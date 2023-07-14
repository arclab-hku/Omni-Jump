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
python train.py --task=aliengo  --num_envs=2000 --headless --seed=${SEED} \
--algo=EST \
--priv_info \
--output_name=aliengo/"${CACHE}" \
${EXTRA_ARGS}
