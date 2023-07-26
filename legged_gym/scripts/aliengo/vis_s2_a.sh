#!/bin/bash
CACHE=$1
python play.py --task=aliengo --s_flag=2 \
--algo=ProprioAdapt \
--priv_info --proprio_adapt \
--output_name=aliengo/"${CACHE}" \
--checkpoint_model=last.pt \
--export_policy