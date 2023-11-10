#!/bin/bash
CACHE=$1
python play.py --task=aliengo --s_flag=1 \
--algo=GenHis \
--priv_info \
--output_name=aliengo/gen_his/"${CACHE}" \
--checkpoint_model=model_2800.pt \
--export_policy \
--export_onnx_policy




