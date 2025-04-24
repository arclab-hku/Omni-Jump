#!/bin/bash

SCRIPT="/home/leo/research/leggedR/adaptive_legged_gym/legged_gym/scripts/aliengo/gen_his/train.sh"

# Number of times to execute the script
TIMES=5

for ((i = 1; i <= TIMES; i++)); do
    PARAM="$i"
    OUTPUT_FILE="Ablation_WO_HR_${i}"
    echo "Execution $i"
    bash "$SCRIPT" "0" "$PARAM" "$OUTPUT_FILE"
done