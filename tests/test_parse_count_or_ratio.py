# tests/test_parse_count_or_ratio.py

import math
import pytest

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# テスト対象をインポート
# 例: あなたのファイル名が active_loop.py の場合
from train_al import parse_count_or_ratio

# 今はサンプルとして関数をここに貼っておきます。
# 実際は上の import を使い、この定義は削除してください。
# def parse_count_or_ratio(spec: str, total: int) -> int:
#     s = str(spec).strip()
#     if s.endswith('%'):
#         p = float(s[:-1]) / 100.0
#         return max(1, int(round(total * p)))
#     try:
#         v = float(s)
#         if 0.0 < v <= 1.0:
#             return max(1, int(round(total * v)))
#         else:
#             return max(1, int(round(v)))
#     except ValueError:
#         raise ValueError(f"Invalid spec '{spec}'. Use forms like '5%', '0.05', or '1000'.")

@pytest.mark.parametrize(
    "spec,total,expected",
    [
        # パーセンテージ表記
        ("5%", 1000, 50),
        ("5%", 123, 6),         # round(6.15)=6
        ("  5% ", 200, 10),     # 前後空白もOK

        # 比率(0~1)
        ("0.05", 1000, 50),
        ("0.05", 123, 6),       # round(6.15)=6
        ("  0.05  ", 200, 10),

        # 絶対件数
        ("1000", 999999, 1000),
        ("  7  ", 1000, 7),
        ("1", 1, 1),
    ],
)
def test_parse_count_or_ratio_happy_path(spec, total, expected):
    assert parse_count_or_ratio(spec, total) == expected

def test_parse_count_or_ratio_min_clip():
    # 0 や 0% のような指定は最小1にクリップされる（現在の実装仕様）
    assert parse_count_or_ratio("0.0", 1000) == 1
    assert parse_count_or_ratio("0%", 1000) == 1
    assert parse_count_or_ratio("0", 1000) == 1

def test_parse_count_or_ratio_large_absolute_ok():
    # total を超える値でも関数内ではクリップしない（呼び出し側でクリップ想定）
    assert parse_count_or_ratio("100000", 1000) == 100000

@pytest.mark.parametrize("bad", ["abc", "", " ", "%", "five%", "1.2.3"])
def test_parse_count_or_ratio_invalid_raises(bad):
    with pytest.raises(ValueError):
        parse_count_or_ratio(bad, 1000)
