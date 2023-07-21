#!/bin/bash
CACHE=$1
python play.py --task=aliengo_rough --s_flag=1 \
--algo=PPO \
--priv_info \
--output_name=aliengo_test/rma/"${CACHE}" \
--checkpoint_model=last.pt \
--export_policy




