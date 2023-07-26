#!/bin/bash
CACHE=$1
python play_joy_45.py --task=go1 --s_flag=1 \
--algo=EST \
--priv_info \
--output_name=go1/"${CACHE}" \
--checkpoint_model=last.pt \
--export_policy




