#!/bin/bash
CACHE=$1
python play.py --task=aliengo_rough --s_flag=1 \
--algo=On \
--priv_info \
--output_name=aliengo_test/oracle/"${CACHE}" \
--checkpoint_model=last.pt \
--export_policy




