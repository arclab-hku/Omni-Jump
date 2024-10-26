#!/bin/bash
CACHE=$1
python play_joy_45.py --task=go2 --s_flag=1 \
--algo=GenHis \
--priv_info \
--output_name=go2/gen_his/"${CACHE}" \
--checkpoint_model=model_1000.pt \
--export_policy




