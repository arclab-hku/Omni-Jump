#!/bin/bash
CACHE=$1
python play.py --task=go1 --s_flag=1 \
--algo=Gen \
--priv_info \
--output_name=go1/gen/"${CACHE}" \
--checkpoint_model=last.pt \
--export_policy




