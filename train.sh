#!/usr/bin/env sh
exp="default"
gpu_num="4"

model="r50_mavos" #ResNet-50 backbone

# model="mavos" #MobileNetV2 backbone
# model="swinb_mavos" #Swin-Base backbone


## Pre-training ##
# We use the same pre-trained weights of DeAoT-L. Download the pre-trained weights on static images (PRE) from the model zoo and place it as result/default_R50_MAVOS/PRE/ema_ckpt/save_step_100000.pth.

# Alternatively, you can pre-train from scratch with the following command:
#stage="pre"
#python tools/train.py --amp \
#    --exp_name ${exp} \
#    --stage ${stage} \
#    --model ${model} \
#    --gpu_num ${gpu_num}


## Training ##
stage="pre_ytb_dav"
python tools/train.py --amp \
	--exp_name ${exp} \
	--stage ${stage} \
	--model ${model} \
	--gpu_num ${gpu_num}
