#!/bin/bash
CACHE=$1
python play.py --task=go2 --s_flag=1 \
--algo=GenHis \
--priv_info \
--output_name=go2/gen_his/"${CACHE}" \
--checkpoint_model=last.pt \
--export_policy \
--export_onnx_policy