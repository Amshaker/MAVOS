import torch
import torch.nn as nn

TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])
if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs, int_classes
else:
    import collections.abc as container_abcs

from timm.models.layers import DropPath, trunc_normal_


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class JointFocalAttention(nn.Module):
    def __init__(self, dim, focal_window, focal_level, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 focal_factor=2, bias=True, use_postln_in_modulation=False, normalize_modulator=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.use_postln_in_modulation = use_postln_in_modulation
        self.normalize_modulator = normalize_modulator

        self.f_context = nn.Linear(dim, dim + (self.focal_level + 1), bias=bias)
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=bias)

        self.act = nn.GELU()
        self.focal_layers = nn.ModuleList()

        self.kernel_sizes = []
        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1,
                              groups=dim, padding=kernel_size // 2, bias=False),
                    nn.GELU(),
                )
            )
            self.kernel_sizes.append(kernel_size)
        if self.use_postln_in_modulation:
            self.ln = nn.LayerNorm(dim)
        
        
    def forward(self, latents, context, H, W):
        # attention part
        B_l, N_l, C_l = latents.shape
        B_c, N_c, C_c = context.shape

        q = self.q(latents).reshape(B_l, N_l, 1, self.num_heads, C_l // self.num_heads).permute(2, 0, 3, 1, 4)
        k = self.k(context).reshape(B_c, N_c, 1, self.num_heads, C_c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k = q[0], k[0]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # focal modulation part
        # pre linear projection
        f_context_out = self.f_context(context.reshape(B_c, H, W, C_c)).permute(0, 3, 1, 2).contiguous()
        v_ctx, self.gates = torch.split(f_context_out, (C_c, self.focal_level + 1), 1)

        # context aggreation
        v_ctx_all = 0
        for l in range(self.focal_level):
            v_ctx = self.focal_layers[l](v_ctx)
            v_ctx_all = v_ctx_all + v_ctx * self.gates[:, l:l + 1]
        v_ctx_global = self.act(v_ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        v_ctx_all = v_ctx_all + v_ctx_global * self.gates[:, self.focal_level:]

        # normalize context
        if self.normalize_modulator:
            v_ctx_all = v_ctx_all / (self.focal_level + 1)

        # focal modulation
        self.v_modulator = self.h(v_ctx_all)
        
        # Back to attention
        v = self.v_modulator.reshape(B_c, N_c, 1,  self.num_heads, C_c // self.num_heads).permute(2, 0, 3, 1, 4)
        v=v[0]
        latents = (attn @ v).transpose(1, 2).reshape(B_l, N_l, C_l)
        latents = self.proj(latents)
        latents = self.proj_drop(latents)
        return latents, attn



class JointBlock(nn.Module):
    def __init__(self, dim, num_heads, focal_level=2, focal_window=3, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 focal_factor=2, bias=True, use_postln_in_modulation=False, normalize_modulator=False):
        super().__init__()
        self.norm_l = norm_layer(dim)
        self.norm_c = norm_layer(dim)
        self.attn = JointFocalAttention(dim, focal_window, focal_level, num_heads, qkv_bias, qk_scale, attn_drop, drop,
                                        focal_factor, bias, use_postln_in_modulation, normalize_modulator)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, latents, context, H, W, return_attention=False):
        #cls_tokens = latents[:, 0].unsqueeze(1)
        #latents = latents[:, 1:]

        y, attn = self.attn(self.norm_l(latents), self.norm_c(context), H, W)
        if return_attention:
            return attn
        latents = latents + self.drop_path(y)
        latents = latents + self.drop_path(self.mlp(self.norm2(latents)))
        #latents = torch.cat((cls_tokens, latents), dim=1)

        return latents



class MemoryJointModulation(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 focal_level=2, focal_window=3, focal_factor=2, bias=True,
                 use_postln_in_modulation=False, normalize_modulator=False
                 ):
        super().__init__()

        # Define a cross focal modulation
        self.joint_modulation = JointBlock(dim, num_heads, focal_level, focal_window, mlp_ratio, qkv_bias, qk_scale,
                                           drop, attn_drop, drop_path, act_layer, norm_layer, focal_factor, bias,
                                           use_postln_in_modulation, normalize_modulator)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, latents, context, H, W):
        latents = latents.permute(1, 0, 2)
        context = context.permute(1, 0, 2)

        out = self.joint_modulation(latents, context, H, W)

        return out.permute(1, 0, 2)


