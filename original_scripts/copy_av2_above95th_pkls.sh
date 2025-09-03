#!/bin/bash
# 95%タイル超えサンプルのpklファイルを抽出してコピーするシェルスクリプト

# 必要に応じてパスを修正してください
python original_code/copy_above95th_pkls.py \
  --csv_path output/av2/high_samples_above_95th.csv \
  --output_dir data_av2/above95th_features/val/ \
  --src_dir data_av2/features/val/ \
  --mode val
