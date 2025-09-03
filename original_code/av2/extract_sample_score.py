import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, required=True, help='評価指標CSVファイルのパス')
    parser.add_argument('--output_high_csv', type=str, default=None, help='95%タイル超えサンプル出力CSV')
    parser.add_argument('--output_low_csv', type=str, default=None, help='5%タイル未満サンプル出力CSV')
    return parser.parse_args()

def main():
    args = parse_arguments()
    csv_path = args.input_csv
    output_high_csv = args.output_high_csv or os.path.join(os.path.dirname(csv_path), 'high_samples_above_95th.csv')
    output_low_csv = args.output_low_csv or os.path.join(os.path.dirname(csv_path), 'low_samples_below_5th.csv')

    # ファイル存在確認
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSVファイルが見つかりません: {csv_path}")

    # CSV 読み込み
    df = pd.read_csv(csv_path)

    # 確認
    print(df.head())
    print(df.info())

    metrics = ['minADE_k', 'minFDE_k', 'b-minFDE_k']

    # 95% と 5% タイルの閾値を計算
    quantiles_95 = {m: df[m].quantile(0.95) for m in metrics}
    quantiles_05 = {m: df[m].quantile(0.05) for m in metrics}

    # すべてのメトリクスで 95% タイル超え
    mask_high = (df[metrics[0]] > quantiles_95[metrics[0]]) & \
                (df[metrics[1]] > quantiles_95[metrics[1]]) & \
                (df[metrics[2]] > quantiles_95[metrics[2]])

    # すべてのメトリクスで 5% タイル以下
    mask_low = (df[metrics[0]] < quantiles_05[metrics[0]]) & \
               (df[metrics[1]] < quantiles_05[metrics[1]]) & \
               (df[metrics[2]] < quantiles_05[metrics[2]])

    # 該当する sample_ID を取得
    high_samples = df.loc[mask_high, "sample_ID"].tolist()
    low_samples = df.loc[mask_low, "sample_ID"].tolist()

    print("95%タイルを超えるサンプル数:", len(high_samples))
    # print("sample_IDs:", high_samples)

    print("5%タイル未満のサンプル数:", len(low_samples))
    # print("sample_IDs:", low_samples)

    # 抽出
    high_samples_df = df.loc[mask_high, ["sample_ID"] + metrics]
    low_samples_df = df.loc[mask_low, ["sample_ID"] + metrics]

    # CSVに書き出し
    high_samples_df.to_csv(output_high_csv, index=False)
    low_samples_df.to_csv(output_low_csv, index=False)

    print(f"95%タイル超えサンプルを {output_high_csv} に保存しました")
    print(f"5%タイル未満サンプルを {output_low_csv} に保存しました")

if __name__ == "__main__":
    main()