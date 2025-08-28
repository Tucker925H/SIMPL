echo "-- Processing val set..."
python data_argo/run_preprocess.py --mode val \
  --data_dir ~/data/dataset/argo_motion_forecasting/val/data/ \
  --save_dir data_argo/features/ \
  --small
  # --debug --viz

echo "-- Processing train set..."
python data_argo/run_preprocess.py --mode train \
  --data_dir ~/data/dataset/argo_motion_forecasting/train/data/ \
  --save_dir data_argo/features/ \
  --small

# echo "-- Processing test set..."
# python data_argo/run_preprocess.py --mode test \
#   --data_dir ~/data/dataset/argo_motion_forecasting/test_obs/data/ \
#   --save_dir data_argo/features/ \
#   --small


# このファイルは --data_dir から1024のサンプル(-smallオプション)を読み込んでデータの前処理として特徴量を抽出し.pklファイルに保存するスクリプトです。