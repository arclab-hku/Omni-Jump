#!/bin/bash
GPUS=$1
SEED=$2
CACHE=$3
PRETRAIN=$4

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

echo extra "${EXTRA_ARGS}"

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py --task=go1  --num_envs=4096 --headless --seed=${SEED} \
--algo=GenHis \
--priv_info \
--output_name=go1/gen_his/"${CACHE}" \
--checkpoint_model=outputs/go1/gen_his/"${PRETRAIN}"/stage1_nn/last.pt \
${EXTRA_ARGS}