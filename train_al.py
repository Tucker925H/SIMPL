import os
import sys
import time
import subprocess
from typing import Any, Dict, List, Tuple, Union
from datetime import datetime
import argparse
import faulthandler
from tqdm import tqdm
#
import torch
import torch.multiprocessing
from torch.utils.data import DataLoader, Subset
#
from loader import Loader
from utils.logger import Logger
from utils.utils import AverageMeterForDict
from utils.utils import save_ckpt, set_seed
import random
import math


def parse_count_or_ratio(spec: str, total: int) -> int:
    """
    "5%", "0.05", "1000" の3形式を受け付け、整数件数を返す。
    - 末尾%: パーセンテージ
    - 0~1の小数: 比率
    - それ以外: 絶対件数
    """
    s = str(spec).strip()
    if s.endswith('%'):
        p = float(s[:-1]) / 100.0
        return max(1, int(round(total * p)))
    try:
        v = float(s)
        if 0.0 < v <= 1.0:
            return max(1, int(round(total * v)))
        else:
            return max(1, int(round(v)))
    except ValueError:
        raise ValueError(f"Invalid spec '{spec}'. Use forms like '5%', '0.05', or '1000'.")


def parse_arguments() -> Any:
    """Arguments for running the baseline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train", type=str, help="Mode, train/val/test")
    parser.add_argument("--features_dir", required=True, default="", type=str, help="Path to the dataset")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, default=16, help="Val batch size")
    parser.add_argument("--train_epoches", type=int, default=10, help="Number of epoches for training")
    parser.add_argument("--val_interval", type=int, default=5, help="Validation intervals")
    parser.add_argument("--data_aug", action="store_true", help="Enable data augmentation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_cuda", action="store_true", help="Use CUDA for acceleration")
    parser.add_argument("--logger_writer", action="store_true", help="Enable tensorboard")
    parser.add_argument("--adv_cfg_path", required=True, default="", type=str)
    parser.add_argument("--rank_metric", required=False, type=str, default="brier_fde_k", help="Ranking metric")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--no_pbar", action="store_true", help="Hide progress bar")
    parser.add_argument("--model_path", required=False, type=str, help="path to the saved model")

    # ★ 追加: 初期ラベル付きデータの指定（比率 or 件数）。例: "5%" / "0.05" / "1000"
    parser.add_argument("--init_labeled", type=str, default="5%",
                        help="Initial labeled amount: '5%%' / '0.05' / '1000' / 'auto' (with --active_rounds).")
    
    # ★ 追加: 各ラウンドで追加する難例の件数（比率 or 件数）。例: "1000" / "2%" / "0.02"
    parser.add_argument("--select_top", type=str, default="10000",
                        help="High-loss picks per round: '1000' / '2%%' / '0.02' / 'auto' (with --active_rounds).")

    # ★ 追加: 総ラウンド数（初回学習も含む）。0 or 未指定で無効。
    parser.add_argument("--active_rounds", type=int, default=0,
                        help="Total Active Learning rounds (including the initial one). If >0 and "
                            "--select_top='auto', per-round picks are computed to finish in exactly this many rounds.")
    
    return parser.parse_args()


def build_dataloader(dataset, batch_size, shuffle, collate_fn, workers=8, drop_last=True, pin_memory=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_fn,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )


def train_one_model(args, device, logger, date_str, net, loss_fn, optimizer, evaluator,
                    train_dataset, val_dataset, net_name, rank_metric,
                    cumulative_start_time: float = None, round_id: int = None):
    """
    既存の「学習ロジック」をそのまま使うための関数化。
    学習内側の処理（for epoch in range...）は一切変更しない。
    """
    dl_train = build_dataloader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=train_dataset.dataset.collate_fn if isinstance(train_dataset, Subset) else train_dataset.collate_fn,
        workers=8, drop_last=True, pin_memory=True
    )
    dl_val = build_dataloader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        workers=8, drop_last=True, pin_memory=True
    )

    niter = 0
    best_metric = 1e6

    for epoch in range(args.train_epoches):
        logger.print('\nEpoch {}'.format(epoch))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # * Train
        epoch_start = time.time()
        train_loss_meter = AverageMeterForDict()
        train_eval_meter = AverageMeterForDict()
        net.train()
        for i, data in enumerate(tqdm(dl_train, disable=args.no_pbar, ncols=80)):
            data_in = net.pre_process(data)
            out = net(data_in)
            loss_out = loss_fn(out, data)

            post_out = net.post_process(out)
            eval_out = evaluator.evaluate(post_out, data)

            optimizer.zero_grad()
            loss_out['loss'].backward()
            lr = optimizer.step()

            train_loss_meter.update(loss_out)
            train_eval_meter.update(eval_out)
            niter += args.train_batch_size
            logger.add_dict(loss_out, niter, prefix='train/')

        optimizer.step_scheduler()
        max_memory = torch.cuda.max_memory_allocated(device=device) // 2 ** 20

        loss_avg = train_loss_meter.metrics['loss'].avg
        logger.print('[Training] Avg. loss: {:.6}, time cost: {:.3} mins, lr: {:.3}, peak mem: {} MB'.
                    format(loss_avg, (time.time() - epoch_start) / 60.0, lr, max_memory))
        logger.print('-- ' + train_eval_meter.get_info())

        logger.add_scalar('train/lr', lr, it=epoch)
        logger.add_scalar('train/max_mem', max_memory, it=epoch)
        for key, elem in train_eval_meter.metrics.items():
            logger.add_scalar(title='train/{}'.format(key), value=elem.avg, it=epoch)

        if ((epoch + 1) % args.val_interval == 0) or epoch > int(args.train_epoches / 2):
            # * Validation
            with torch.no_grad():
                val_start = time.time()
                val_loss_meter = AverageMeterForDict()
                val_eval_meter = AverageMeterForDict()
                net.eval()
                for i, data in enumerate(tqdm(dl_val, disable=args.no_pbar, ncols=80)):
                    data_in = net.pre_process(data)
                    out = net(data_in)
                    loss_out = loss_fn(out, data)

                    post_out = net.post_process(out)
                    eval_out = evaluator.evaluate(post_out, data)

                    val_loss_meter.update(loss_out)
                    val_eval_meter.update(eval_out)

                logger.print('[Validation] Avg. loss: {:.6}, time cost: {:.3} mins'.format(
                    val_loss_meter.metrics['loss'].avg, (time.time() - val_start) / 60.0))
                logger.print('-- ' + val_eval_meter.get_info())

                for key, elem in val_loss_meter.metrics.items():
                    logger.add_scalar(title='val/{}'.format(key), value=elem.avg, it=epoch)
                for key, elem in val_eval_meter.metrics.items():
                    logger.add_scalar(title='val/{}'.format(key), value=elem.avg, it=epoch)

                if (epoch >= args.train_epoches / 2):
                    if val_eval_meter.metrics[rank_metric].avg < best_metric:
                        model_name = date_str + '_{}_best.tar'.format(net_name)
                        save_ckpt(net, optimizer, epoch, 'saved_models/', model_name)
                        best_metric = val_eval_meter.metrics[rank_metric].avg
                        print('Save the model: {}, {}: {:.4}, epoch: {}'.format(
                            model_name, rank_metric, best_metric, epoch))

        if int(100 * epoch / args.train_epoches) in [20, 40, 60, 80]:
            model_name = date_str + '_{}_ckpt_epoch{}.tar'.format(net_name, epoch)
            save_ckpt(net, optimizer, epoch, 'saved_models/', model_name)
            logger.print('Save the model to {}'.format('saved_models/' + model_name))

    # Compute elapsed time. If a cumulative start time is provided (from main),
    # report cumulative time up to this round; otherwise report local training time.
    end_time = time.time()
    try:
        if cumulative_start_time is not None:
            elapsed_min = (end_time - cumulative_start_time) / 60.0
        else:
            elapsed_min = (end_time - epoch_start) / 60.0
    except Exception:
        elapsed_min = (end_time - start_time) / 60.0 if 'start_time' in locals() else 0.0

    logger.print(f"\nTraining completed in {elapsed_min:.2f} mins at round {round_id if round_id is not None else 'N/A'}")

    # save trained model (final for this round)
    model_name = date_str + '_{}_epoch{}.tar'.format(net_name, args.train_epoches)
    save_ckpt(net, optimizer, epoch, 'saved_models/', model_name)
    print('Save the model to {}'.format('saved_models/' + model_name))


def compute_pool_losses(args, device, net, loss_fn, pool_subset, batch_size, disable_pbar):
    dl_pool = DataLoader(
        pool_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=pool_subset.dataset.collate_fn,
        drop_last=False,
        pin_memory=True,
    )
    per_sample_losses: List[float] = []
    net.eval()
    with torch.no_grad():
        for data in tqdm(dl_pool, disable=disable_pbar, ncols=80):
            data_in = net.pre_process(data)
            out = net(data_in)
            # ここでreturn_per_sample=Trueを渡す
            loss_out = loss_fn(out, data)
            # サンプルごとの損失を取得
            if 'loss_per_sample' in loss_out:
                vals = loss_out['loss_per_sample']
                if isinstance(vals, torch.Tensor):
                    vals = vals.detach().cpu().float().tolist()
                else:
                    vals = list(vals)
            else:
                # フォールバック: バッチ平均
                # bsz = next(iter(data.values())).shape[0]
                avg_loss = float(loss_out['loss'].detach().cpu())
            per_sample_losses.extend([avg_loss])
    return per_sample_losses


def main():
    args = parse_arguments()

    faulthandler.enable()
    start_time = time.time()
    set_seed(args.seed)

    if args.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda", 0)
    else:
        device = torch.device('cpu')

    print('GPU Device: {}'.format(device))
    # sys.exsit(0)  # DEBUG

    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "log/" + date_str
    logger = Logger(date_str=date_str, log_dir=log_dir, enable_flags={'writer': args.logger_writer})
    logger.log_basics(args=args, datetime=date_str)

    # ★ 初回ロード（データセットやネット構成名取得用）
    base_loader = Loader(args, device, is_ddp=False)
    if args.resume:
        logger.print('[Resume] Loading state_dict from {}'.format(args.model_path))
        base_loader.set_resmue(args.model_path)
    (train_set, val_set), _, _, _, _ = base_loader.load()
    
    net_name = base_loader.network_name()
    rank_metric = args.rank_metric

    # ★ インデックス分割（初期5%＋残り）
    total = len(train_set)
    all_indices = list(range(total))

    # ★ 使用する割合を変数で指定（例：0.05 → 5%）
    train_ratio = 0.01

    # 使用するデータ数を計算
    # subset_size = int(total * train_ratio)
    subset_size = int(total)

    logger.print(f"[ActiveLoop] Using {subset_size} samples.")
    
    # random.Random(args.seed).shuffle(all_indices)
    # シード付き乱数生成器を作成
    rng = random.Random(args.seed)

    # 重複なしでランダムサンプリング
    all_indices = rng.sample(all_indices, subset_size)
    logger.print(f"[ActiveLoop] Sampled {len(all_indices)} unique indices.")

    # init_labeled の決定
    if isinstance(args.init_labeled, str) and args.init_labeled.lower() == "auto" and args.active_rounds > 0:
        init_k = math.ceil(subset_size / args.active_rounds)  # N分割の1ラウンド目
    else:
        init_k = parse_count_or_ratio(args.init_labeled, subset_size)

    init_k = min(max(1, init_k), subset_size)
    labeled_indices = all_indices[:init_k]
    pool_indices = all_indices[init_k:]

    logger.print(f"[ActiveLoop] total={total}, init_labeled={len(labeled_indices)}, pool={len(pool_indices)}")
    # sys.exit(0)  # DEBUG

    round_id = 0
    used_mask = [False] * total
    for idx in labeled_indices:
        used_mask[idx] = True

    while True:
        round_id += 1
        logger.print(f"\n========== Active Learning Round {round_id} ==========")
        logger.print(f"[Round {round_id}] labeled={len(labeled_indices)}, pool={len(pool_indices)}")

        # ★ モデルを0から再構築（毎ラウンド初期化）
        loader = Loader(args, device, is_ddp=False)
        (train_set, val_set), net, loss_fn, optimizer, evaluator = loader.load()

        # ★ 今ラウンドの学習用データは「累積の labeled_indices 」
        train_subset = Subset(train_set, sorted(labeled_indices))

        # ★ 学習（エポック内のロジックは既存コードそのま）
        train_one_model(
            args=args, device=device, logger=logger, date_str=date_str,
            net=net, loss_fn=loss_fn, optimizer=optimizer, evaluator=evaluator,
            train_dataset=train_subset, val_dataset=val_set,
            net_name=net_name, rank_metric=rank_metric,
            cumulative_start_time=start_time, round_id=round_id
        )


        # ★ プールが空なら終了
        if not pool_indices:
            logger.print("[ActiveLoop] pool empty -> exit")
            break

        # ★ プール上で推論して high-loss サンプルを選ぶ
        pool_subset = Subset(train_set, pool_indices)
        per_sample_losses = compute_pool_losses(
            args=args, device=device, net=net, loss_fn=loss_fn,
            pool_subset=pool_subset, batch_size=1,
            disable_pbar=args.no_pbar
        )
        print(f"Computed losses for {len(per_sample_losses)} samples in pool.")
        print(f"pool_indices: {len(pool_indices)} ...")
        print(f"pool_subset: {len(pool_subset)} ...")
        # sys.exit(0)
        assert len(per_sample_losses) == len(pool_indices), "Losses and pool indices must align."

        # ★ 追加件数（比率 or 件数）を決定
        # 追加件数の決定
        if isinstance(args.select_top, str) and args.select_top.lower() == "auto" and args.active_rounds > 0:
            rounds_left = args.active_rounds - round_id  # いまが round_id（1始まり）なので、残りラウンド数
            if rounds_left <= 0:
                select_k = len(pool_indices)  # 念のため：最終は全て
            else:
                # 残りを等分割（切り上げ）して、このラウンドで取る
                select_k = math.ceil(len(pool_indices) / rounds_left)
        else:
            select_k = parse_count_or_ratio(args.select_top, len(pool_indices))

        select_k = min(max(1, select_k), len(pool_indices))

        # ★ 上位 select_k 件の「loss が大きい」インデックスを抽出
        #    per_sample_losses は pool_indices と同順なので、ソートは (loss, idx_in_pool) で行う
        ranked = sorted(
            enumerate(per_sample_losses),
            key=lambda x: x[1],
            reverse=True
        )
        take_pool_positions = [pos for (pos, _) in ranked[:select_k]]
        newly_selected = [pool_indices[pos] for pos in take_pool_positions]

        # ★ 累積ラベル集合に追加
        labeled_indices.extend(newly_selected)
        for idx in newly_selected:
            used_mask[idx] = True

        # ★ プールから除外
        keep_mask = [True] * len(pool_indices)
        for pos in take_pool_positions:
            keep_mask[pos] = False
        pool_indices = [idx for keep, idx in zip(keep_mask, pool_indices) if keep]

        logger.print(f"[Round {round_id}] add {len(newly_selected)} hard samples -> labeled={len(labeled_indices)}, pool={len(pool_indices)}")

        # ★ すべて使い切ったら終了
        if not pool_indices:
            logger.print("[ActiveLoop] pool empty after selection -> exit")

    elapsed_min = (time.time() - start_time) / 60.0
    print(f'\nExit... (total time: {elapsed_min:.2f} mins)\n')


if __name__ == "__main__":
    main()
