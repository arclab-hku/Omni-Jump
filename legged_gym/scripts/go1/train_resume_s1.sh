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
python train.py --task=aliengo  --num_envs=4096 --headless --seed=${SEED} \
--s_flag=1 \
--algo=On \
--priv_info \
--output_name=aliengo/base_test/"${CACHE}" \
--checkpoint_model=model_800.pt \
${EXTRA_ARGS}
