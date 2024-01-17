#!/bin/bash
CACHE=$1
python play_joy_45.py --task=arcdog --s_flag=1 \
--algo=GenHis \
--priv_info \
--output_name=arcdog/gen_his/"${CACHE}" \
--checkpoint_model=last.pt \
--export_policy




