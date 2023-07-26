#!/bin/bash
GPUS=$1
SEED=$2
CACHE=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

echo extra "${EXTRA_ARGS}"

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py --task=go1  --num_envs=4096 --headless --seed=${SEED} \
--s_flag=2 \
--algo=ProprioAdapt \
--priv_info --proprio_adapt \
--output_name=go1/"${CACHE}" \
--checkpoint_model=7000m.pt \
${EXTRA_ARGS}
