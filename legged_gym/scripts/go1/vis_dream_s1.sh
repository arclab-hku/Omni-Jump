#!/bin/bash
CACHE=$1
python play.py --task=go1 --s_flag=1 \
--algo=Dream \
--priv_info \
--output_name=go1_plane/dream/"${CACHE}" \
--checkpoint_model=last.pt \
--export_policy




