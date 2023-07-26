#!/bin/bash
CACHE=$1
python play_joy_45.py --task=go1 --s_flag=1 \
--algo=On \
--priv_info \
--output_name=go1/oracle/"${CACHE}" \
--checkpoint_model=last.pt \
--export_policy




