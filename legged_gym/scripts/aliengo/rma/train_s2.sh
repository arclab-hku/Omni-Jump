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
python train.py --task=aliengo --headless --seed=${SEED} \
--num_envs=5000 \
--algo=ProprioAdapt \
--priv_info --proprio_adapt \
--output_name=aliengo/rma/"${CACHE}" \
--checkpoint_model=outputs/aliengo/rma/"${CACHE}"/stage1_nn/last.pt \
${EXTRA_ARGS}
