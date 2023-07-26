#!/bin/bash
CACHE=$1
python play.py --task=go1 --s_flag=1 \
--algo=Our \
--priv_info \
--output_name=go1/our/"${CACHE}" \
--checkpoint_model=last.pt \
--export_policy




