import os
import shutil
import pandas as pd
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True, type=str, help="データセットのパス")
    parser.add_argument("--src_dir", required=True, type=str, help="元のpklファイルがあるディレクトリ")
    parser.add_argument("--mode", default="val", type=str, help="モード: train/val/test")
    parser.add_argument("--output_dir", default="data_argo/above95th_features/val/", type=str, help="出力ディレクトリ")
    return parser.parse_args()


def main():
    arg = parse_arguments()
    # 出力ディレクトリを再帰的に作成（既に存在してもOK）
    os.makedirs(arg.output_dir, exist_ok=True)

    # sample_IDリスト取得
    high_samples_df = pd.read_csv(arg.csv_path)
    sample_ids = high_samples_df["sample_ID"].astype(str).tolist()
    
    # コピー処理
    copied = 0
    not_found = 0
    for sid in sample_ids:
        src_pkl = os.path.join(arg.src_dir, f"{sid}.pkl")
        dst_pkl = os.path.join(arg.output_dir, f"{sid}.pkl")
        if os.path.exists(src_pkl):
            shutil.copy2(src_pkl, dst_pkl)
            copied += 1
        else:
            print(f"Not found: {src_pkl}")
            not_found += 1
    
    print(f"コピー完了: {copied}件, 見つからなかったファイル: {not_found}件")


if __name__ == "__main__":
    main()
