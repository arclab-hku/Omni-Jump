#!/bin/bash
CACHE=$1
python play.py --task=aliengo_rough \
--algo=On \
--priv_info \
--output_name=alieng_test/"${CACHE}" \
--checkpoint_model=model_800.pt \
#--export_policy




