#!/bin/bash
CACHE=$1
python play.py --task=aliengo_rough --s_flag=1 \
--algo=Dream \
--priv_info \
--output_name=alieng_test/dream/"${CACHE}" \
--checkpoint_model=last.pt \
--export_policy




