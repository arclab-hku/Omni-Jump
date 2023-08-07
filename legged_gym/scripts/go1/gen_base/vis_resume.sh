#!/bin/bash
CACHE=$1
RESUMENAME=$2

python play.py --task=go1 --s_flag=3 \
--algo=GenBase \
--priv_info \
--output_name=go1/gen_base/"${CACHE}" \
--checkpoint_model=last.pt \
--resume \
--resume_name="${RESUMENAME}" \
--export_policy





