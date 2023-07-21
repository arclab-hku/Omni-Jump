#!/bin/bash
CACHE=$1
python play_joy_45.py --task=aliengo_rough --s_flag=1 \
--algo=Our \
--priv_info \
--output_name=aliengo_test/our/"${CACHE}" \
--checkpoint_model=model_1000.pt \
--export_policy




