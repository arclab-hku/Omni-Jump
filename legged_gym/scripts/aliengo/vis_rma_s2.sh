#!/bin/bash
CACHE=$1
python play_joy_45.py --task=aliengo_rough --s_flag=2 \
--algo=ProprioAdapt \
--priv_info --proprio_adapt \
--output_name=aliengo_test/rma/"${CACHE}" \
--checkpoint_model=best.pt \
--export_policy