#!/bin/bash
CACHE=$1
python play.py --task=go1 --s_flag=1 \
--algo=GenHis \
--priv_info \
--output_name=go1/gen_his/"${CACHE}" \
--checkpoint_model=model_3000.pt \
--export_policy \
--export_onnx_policy




