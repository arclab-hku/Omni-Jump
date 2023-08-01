#!/bin/bash
CACHE=$1
python play_joy_45.py --task=aliengo --s_flag=1 \
--algo=ESTRough \
--priv_info \
--output_name=aliengo/est_rough/"${CACHE}" \
--checkpoint_model=last.pt \
--export_policy




