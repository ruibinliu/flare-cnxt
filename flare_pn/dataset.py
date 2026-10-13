
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import Config

# 数据划分：70% / 15% / 15%
TEST_SIZE_STAGE1 = 0.30
TEST_SIZE_STAGE2 = 0.50

# ============================================================
# 2. 数值特征构造
# ============================================================

# 28 个原始特征内部索引
# 这里的索引是相对于 E-AF 这 28 列，而不是 Excel 全表索引。
RATIO_PAIRS = [
    (1, 0),
    (3, 2),
    (5, 4),
    (7, 6),
    (9, 8),
    (11, 10),
    (13, 12),
    (15, 14),
    (24, 23),
    (26, 25),
]

DIFF_PAIRS = [
    (1, 0),
    (3, 2),
    (5, 4),
    (9, 8),
    (11, 10),
    (13, 12),
    (15, 14),
    (24, 23),
    (26, 25),
]


def build_augmented_features(x_raw):
    """
    输入：
        x_raw: [N, 28]

    输出：
        x_aug: [N, 47]

    组成：
        28 个原始指标 + 10 个 actual/pred ratio + 9 个差值
    """

    ratios = [(x_raw[:, a] / (x_raw[:, b] + 1e-6))[:, None] for a, b in RATIO_PAIRS]

    diffs = [(x_raw[:, a] - x_raw[:, b])[:, None] for a, b in DIFF_PAIRS]

    x_aug = np.concatenate([x_raw] + ratios + diffs, axis=1, ).astype(np.float32)

    return x_aug


def make_ple_edges(x_train_z, n_bins=20):
    """
    基于训练集标准化后的数据计算分位点边界。

    x_train_z:
        [N_train, F]

    返回：
        edges: [F, n_bins + 1]
    """

    q = np.linspace(0.0, 1.0, n_bins + 1)

    edges = np.quantile(x_train_z, q, axis=0).T.astype(np.float32)

    # 某些特征可能存在大量相同值，导致相邻边界相同。
    # 添加极小偏移避免分母为0。
    for j in range(edges.shape[0]):
        for k in range(1, edges.shape[1]):
            if edges[j, k] <= edges[j, k - 1]:
                edges[j, k] = edges[j, k - 1] + 1e-6

    return edges


def ple_encode(x_z, edges):
    """
    分段线性编码。

    输入：
        x_z   : [N, F]
        edges : [F, n_bins + 1]

    输出：
        [N, F, n_bins]
    """

    n_bins = edges.shape[1] - 1

    out = np.empty((len(x_z), x_z.shape[1], n_bins), dtype=np.float32)

    for j in range(x_z.shape[1]):
        xx = x_z[:, j, None]

        left = edges[j, :-1][None, :]
        right = edges[j, 1:][None, :]

        out[:, j, :] = np.clip((xx - left) / (right - left + 1e-8), 0.0, 1.0)

    return out


def load_and_prepare_data(excel_path):
    """
    完整数据预处理

    Excel 格式：
        A列：标签
        B-D：ID
        E-AF：28个数值指标

    返回：
        x_ple, y, split_indices, preprocessing_info
    """

    if not excel_path.exists():
        raise FileNotFoundError(
            f"\n找不到 Excel：{excel_path}\n"
            "请把 FV线性全.xlsx 与本 py 文件放在同一目录，"
            "或放到 ./data/FV线性全.xlsx。\n"
        )

    print(f"读取数据：{excel_path}")

    df = pd.read_excel(excel_path, usecols="A:AF")

    # --------------------------------------------------------
    # 标签
    # --------------------------------------------------------
    label_text = df.iloc[:, 0].astype(str).str.strip().to_numpy()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(label_text).astype(np.int64)

    classes = label_encoder.classes_

    print("类别：")
    for i, c in enumerate(classes):
        print(f"  {i}: {c}")

    # --------------------------------------------------------
    # 分层划分
    # --------------------------------------------------------
    all_idx = np.arange(len(y))

    train_idx, temp_idx = train_test_split(all_idx, test_size=TEST_SIZE_STAGE1, stratify=y, random_state=Config.SEED)

    val_idx, test_idx = train_test_split(temp_idx, test_size=TEST_SIZE_STAGE2, stratify=y[temp_idx], random_state=Config.SEED)

    print(
        f"\n数据划分："
        f"train={len(train_idx)}, "
        f"val={len(val_idx)}, "
        f"test={len(test_idx)}"
    )

    # --------------------------------------------------------
    # E-AF：28个原始数值变量
    # pandas 下标 4:32
    # --------------------------------------------------------
    x_raw = (
        df.iloc[:, 4:32]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float32)
    )

    if x_raw.shape[1] != 28:
        raise ValueError(f"E-AF 应为 28 个数值特征，当前读取到 {x_raw.shape[1]} 个。")

    feature_cols = list(df.columns[4:32])

    # --------------------------------------------------------
    # 缺失值：只使用训练集计算中位数
    # --------------------------------------------------------
    median = np.nanmedian(x_raw[train_idx], axis=0,).astype(np.float32)
    nan_rows, nan_cols = np.where(np.isnan(x_raw))
    x_raw[nan_rows, nan_cols] = median[nan_cols]

    # 处理可能的 inf
    x_raw = np.nan_to_num(x_raw, nan=0.0, posinf=0.0, neginf=0.0)

    # --------------------------------------------------------
    # 28 + 10 + 9 = 47维
    # --------------------------------------------------------
    x_aug = build_augmented_features(x_raw)
    print(f"数值特征：28 raw + 10 ratio + 9 diff = {x_aug.shape[1]} dims")

    # --------------------------------------------------------
    # 标准化：只使用训练集统计量
    # --------------------------------------------------------
    mean = x_aug[train_idx].mean(axis=0).astype(np.float32)
    std = x_aug[train_idx].std(axis=0).astype(np.float32) + 1e-6
    x_z = ((x_aug - mean) / std).astype(np.float32)

    # --------------------------------------------------------
    # PLE边界：只使用训练集
    # --------------------------------------------------------
    edges = make_ple_edges(x_z[train_idx], n_bins=Config.N_BINS)
    x_ple = ple_encode(x_z, edges)
    print(f"PLE tensor shape: {x_ple.shape}")

    preprocessing_info = {
        "classes": classes,
        "feature_cols": feature_cols,
        "median_raw": median,
        "mean_aug": mean,
        "std_aug": std,
        "edges_aug": edges,
        "ratio_pairs": RATIO_PAIRS,
        "diff_pairs": DIFF_PAIRS,
    }

    split_indices = {"train": train_idx, "val": val_idx, "test": test_idx}

    return (x_ple, y, split_indices, preprocessing_info)
# excel_path = Path('data/raw/pulmonary.xlsx')
# dataset = load_and_prepare_data(excel_path=excel_path)
