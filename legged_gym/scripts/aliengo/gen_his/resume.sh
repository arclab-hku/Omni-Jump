#!/bin/bash
GPUS=$1
SEED=$2
CACHE=$3
RESUMENAME=$4

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

echo extra "${EXTRA_ARGS}"

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py --task=aliengo  --num_envs=4096 --headless --seed=${SEED} \
--s_flag=1 \
--algo=GenHis \
--priv_info \
--output_name=aliengo/gen_his/"${CACHE}" \
--checkpoint_model=last.pt \
--resume \
--resume_name="${RESUMENAME}" \
${EXTRA_ARGS}
