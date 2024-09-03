#!/usr/bin/env sh
exp="default"
gpu_num="1"

model="r50_mavos" #ResNet-50 backbone

# model="mavos" #MobileNetV2 backbone
# model="swinb_mavos" #Swin-Base backbone
	

## Evaluation ##

# LVOS Evaluation
stage="pre_ytb_dav_long_lvos"
dataset="lvos"
split="val"
checkpoint="" #Path of trained checkpoint

python tools/eval.py --exp_name ${exp} --stage ${stage} --model ${model} \
	--dataset ${dataset} --split ${split} --gpu_num ${gpu_num} --ckpt_path ${checkpoint}

# Long-Video (LTV) Evaluation
#stage="pre_ytb_dav_long"
#dataset="longvideo"
#split="val"
#python tools/eval.py --exp_name ${exp} --stage ${stage} --model ${model} \
#    --dataset ${dataset} --split ${split} --gpu_num ${gpu_num} --ckpt_path ${checkpoint}

# DAVIS Evaluation
#stage="pre_ytb_dav"
#dataset="davis2017"
#split="val"
#python tools/eval.py --exp_name ${exp} --stage ${stage} --model ${model} \
#    --dataset ${dataset} --split ${split} --gpu_num ${gpu_num} --ckpt_path ${checkpoint}
