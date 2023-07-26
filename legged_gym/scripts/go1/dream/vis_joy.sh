#!/bin/bash
CACHE=$1
python play_joy_45.py --task=go1 --s_flag=1 \
--algo=Dream \
--priv_info \
--output_name=go1/dream/"${CACHE}" \
--checkpoint_model=last.pt \
--export_policy




