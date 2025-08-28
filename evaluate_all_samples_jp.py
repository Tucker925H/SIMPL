import os
import sys
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from importlib import import_module
from torch.utils.data import DataLoader
from loader import Loader

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", required=True, type=str, help="データセットのパス")
    parser.add_argument("--adv_cfg_path", required=True, type=str, help="adv_cfgのパス (例: config.simpl_av2_cfg)")
    parser.add_argument("--model_path", required=True, type=str, help="学習済みモデルのパス")
    parser.add_argument("--data_aug", action="store_true", help="Enable data augmentation")
    parser.add_argument("--mode", default="val", type=str, help="モード: train/val/test")
    parser.add_argument("--visualizer", default="", type=str, help="ビジュアライザーの種類 (未使用)")
    parser.add_argument("--use_cuda", action="store_true", help="CUDAを使用する")
    parser.add_argument("--output_csv", default="eval_metrics.csv", type=str, help="出力CSVファイル名")
    return parser.parse_args()

def main():
    args = parse_arguments()
    if args.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")

    # Loaderの初期化
    loader = Loader(args, device, is_ddp=False)
    loader.set_resmue(args.model_path)
    (train_set, val_set), net, _, _, evaluator = loader.load()
    net.eval()

    # データセットの選択
    if args.mode == "train":
        dataset = train_set
    elif args.mode == "val":
        dataset = val_set
    else:
        dataset = val_set  # fallback

    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=0,
                            collate_fn=dataset.collate_fn,
                            drop_last=False)

    records = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader, desc="評価中")):
            
            # 学習済モデルを使用してデータを前処理
            data_in = net.pre_process(data)
            # モデルにデータを入力して出力を取得
            out = net(data_in)
            # 出力を後処理
            post_out = net.post_process(out)

            # 評価指標を計算
            eval_out = evaluator.evaluate(post_out, data)
            # sample_idと指標を抽出
            sample_id = data["SEQ_ID"][0] if "SEQ_ID" in data else None

            # if i == 1:
            #     print(f"Sample ID: {sample_id}, Evaluation Output: {eval_out}")
            #     print(data["SEQ_ID"])
            #     break

            record = {
                "sample_ID": sample_id,
                "minADE_k": eval_out.get("minade_k", None),
                "minFDE_k": eval_out.get("minfde_k", None),
                "MR_k": eval_out.get("mr_k", None),
                "b-minFDE_k": eval_out.get("brier_fde_k", None),
            }
            records.append(record)

    df = pd.DataFrame(records)
    print("\n評価結果一覧:")
    print(df)
    df.to_csv(args.output_csv, index=False)
    print(f"\n評価指標を {args.output_csv} に保存しました")

if __name__ == "__main__":
    main()
