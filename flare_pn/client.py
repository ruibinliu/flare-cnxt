# -*- coding: utf-8 -*-
"""
ple_node_standalone.py

PLE + NODE 单文件版本
====================

功能：
1. 读取 FV线性全.xlsx
2. 读取 A 列标签、E-AF 共 28 个数值指标
3. 构造 28 原始指标 + 10 比值 + 9 差值 = 47 维数值特征
4. 训练集统计量中位数填补缺失值
5. 训练集均值/标准差标准化
6. 基于训练集分位点构造 20-bin PLE 编码
7. 训练 PLE + NODE 模型
8. 在验证集上早停并保存最佳模型
9. 输出 Validation / Test Accuracy、Balanced Accuracy、Macro-F1、混淆矩阵
10. 保存完整 checkpoint，包含模型参数、类别顺序、预处理统计量与数据划分

运行方式：
    把本文件与 FV线性全.xlsx 放在同一目录，
    然后在 PyCharm 直接点击 Run。

依赖：
    pip install numpy pandas openpyxl scikit-learn torch
"""
from math import ceil
from pathlib import Path
import copy
import random

import numpy as np
import nvflare.client as flare  # 新增
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from config import Config
from flare_pn.dataset import load_and_prepare_data
from flare_pn.model import PLENODE, N_TREES, TREE_DEPTH, TREE_OUT

# ============================================================
# 0. 参数设置
# ============================================================

ROOT = Path(__file__).resolve().parent

# 默认 Excel 与 py 放在同一目录。
# 如果不存在，则自动尝试 ./data/FV线性全.xlsx
# EXCEL_PATH = ROOT / "FV线性全.xlsx"
# if not EXCEL_PATH.exists():
#     EXCEL_PATH = ROOT / "data" / "FV线性全.xlsx"
ROOT = Path(Config.RUNTIME_DATA_ROOT)
EXCEL_PATH = ROOT / "raw" / "pulmonary.xlsx"

MODEL_OUT = ROOT / "ple_node_standalone_best.pt"
RESULT_OUT = ROOT / "ple_node_standalone_test_predictions.csv"

# GPU 可用时自动使用 GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f'DEVICE: {DEVICE}')

# CPU 下限制线程数，避免部分高核心数机器出现线程调度过慢。
CPU_NUM_THREADS = 4
if DEVICE.type == "cpu":
    torch.set_num_threads(CPU_NUM_THREADS)


# ============================================================
# 1. 固定随机种子
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 尽量保证可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False








# ============================================================
# 6. Batch生成
# ============================================================
def batch_indices(indices, batch_size, rng):
    """
    训练集每轮随机打乱。
    """
    perm = rng.permutation(indices)

    for start in range(0, len(perm), batch_size, ):
        yield perm[start:start + batch_size]


# ============================================================
# 7. 模型评估
# ============================================================
@torch.no_grad()
def evaluate(model, x_tensor, y, indices, batch_size=512):
    model.eval()

    all_pred = []
    all_prob = []

    for start in range(0, len(indices), batch_size):
        ids = indices[start:start + batch_size]
        xb = x_tensor[ids].to(DEVICE)
        logits = model(xb)
        prob = torch.softmax(logits, dim=1)
        pred = prob.argmax(dim=1)
        all_pred.append(pred.cpu().numpy())
        all_prob.append(prob.cpu().numpy())

    pred = np.concatenate(all_pred)
    prob = np.concatenate(all_prob)

    y_true = y[indices]

    result = {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
    }

    return result, pred, prob


# ============================================================
# 8. 训练
# ============================================================
def train_model(x_ple, y, split_indices, n_classes):
    train_idx = split_indices["train"]
    val_idx = split_indices["val"]
    test_idx = split_indices["test"]

    # FLARE federated learning
    # flare.init()
    sys_info = flare.system_info()
    client_name = sys_info["site_name"]
    print(f'Flare inited.')

    # 整个 PLE tensor 保留在 CPU。
    # 每个 batch 再送到 GPU，避免显存浪费。
    x_tensor = torch.from_numpy(x_ple).float()
    y_tensor = torch.from_numpy(y).long()

    model = PLENODE(
        n_features=x_ple.shape[1],
        n_bins=x_ple.shape[2],
        emb_dim=Config.EMB_DIM,
        n_classes=n_classes,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    rng = np.random.default_rng(Config.SEED)

    best_state = None
    best_val_accuracy = -1.0
    best_val_result = None
    best_epoch = 0

    no_improve = 0

    print("\n开始训练")
    print(f"Device: {DEVICE}")
    print(f"Max epochs={Config.NUM_EPOCHS}, batch={Config.BATCH_SIZE}, lr={Config.LEARNING_RATE}")
    print("-" * 72)

    while flare.is_running():
        # 1. receive the global model
        input_model = flare.receive()

        # 2. override local model
        model.load_state_dict(input_model.params)

        # 3. evaluate the received global model before local training
        matrics_before_training, pred, prob = evaluate(model, x_tensor, y, test_idx)

        # (optional) Task branch for cross-site evaluation
        if flare.is_evaluate():
            print(f"site = {client_name}, running cross-site evaluation")
            # For CSE, just return the evaluation metrics without training
            output_model = flare.FLModel(metrics=matrics_before_training)
            flare.send(output_model)
            continue

        model.train()
        num_train_batches = ceil(len(train_idx) / Config.BATCH_SIZE)
        steps = Config.NUM_EPOCHS * num_train_batches
        for epoch in range(1, Config.NUM_EPOCHS + 1):
            total_loss = 0.0
            total_num = 0

            for ids in batch_indices(train_idx, Config.BATCH_SIZE, rng):
                xb = x_tensor[ids].to(DEVICE)
                yb = y_tensor[ids].to(DEVICE)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
                optimizer.step()
                total_loss += (loss.item() * len(ids))
                total_num += len(ids)

            train_loss = (total_loss / total_num)

            val_result, _, _ = evaluate(model, x_tensor, y, val_idx)

            print(f"Epoch {epoch:03d} | "
                  f"loss={train_loss:.5f} | "
                  f"val_acc={val_result['accuracy']:.4f} | "
                  f"val_bal_acc={val_result['balanced_accuracy']:.4f} | "
                  f"val_macro_f1={val_result['macro_f1']:.4f}")

            # 以验证集 Accuracy 选择最佳 checkpoint
            if (val_result["accuracy"] > best_val_accuracy):
                best_val_accuracy = (val_result["accuracy"])
                best_val_result = (val_result.copy())
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1

            if (no_improve >= Config.EARLY_STOP_PATIENCE):
                print(f"\nEarly stopping: {Config.EARLY_STOP_PATIENCE} epochs without improvement.")
                break

        if best_state is None:
            raise RuntimeError("训练未获得有效模型。")

        model.load_state_dict(best_state)

        print(f"\nBest epoch: {best_epoch}")

        print(f"Finished Training for {client_name}")
        matrics, pred, prob = evaluate(model, x_tensor, y, val_idx)

        # 5. 发送更新后的模型
        output_model = flare.FLModel(
            params=model.state_dict(),
            metrics=matrics,
            meta={"NUM_STEPS_CURRENT_ROUND": steps},
        )
        flare.send(output_model)

    return (model, x_tensor, best_epoch, best_val_result)


# ============================================================
# 9. 保存测试预测
# ============================================================

def save_test_predictions(path, test_idx, y, pred, prob, classes):
    result = pd.DataFrame({
        "row_index": test_idx,
        "true_id": y[test_idx],
        "pred_id": pred,
        "true_label": [classes[i] for i in y[test_idx]],
        "pred_label": [classes[i] for i in pred],
        "confidence": prob.max(axis=1),
    })

    for i, cls in enumerate(classes):
        result[f"prob_{i}_{cls}"] = prob[:, i]

    result.to_csv(path, index=False, encoding="utf-8-sig")


def save_checkpoint(model, prep, split_indices, best_epoch, val_result, test_result):
    checkpoint = {
        "state_dict": model.state_dict(),
        "classes": prep["classes"],
        "feature_cols": prep["feature_cols"],
        "median_raw": prep["median_raw"],
        "mean_aug": prep["mean_aug"],
        "std_aug": prep["std_aug"],
        "edges_aug": prep["edges_aug"],
        "ratio_pairs": prep["ratio_pairs"],
        "diff_pairs": prep["diff_pairs"],
        "split": split_indices,
        "best_epoch": best_epoch,
        "validation_result": val_result,
        "test_result": test_result,

        "config": {
            "seed": Config.SEED,
            "n_bins": Config.N_BINS,
            "emb_dim": Config.EMB_DIM,
            "n_trees": N_TREES,
            "tree_depth": TREE_DEPTH,
            "tree_out": TREE_OUT,
            "batch_size": Config.BATCH_SIZE,
            "learning_rate": Config.LEARNING_RATE,
            "weight_decay": Config.WEIGHT_DECAY,
            "early_stop_patience": Config.EARLY_STOP_PATIENCE,
        },
    }

    torch.save(checkpoint, MODEL_OUT)


# ============================================================
# 10. 主程序
# ============================================================

def main():
    set_seed(Config.SEED)

    # 数据处理
    flare.init()
    data_file = ROOT / flare.get_site_name() / "data.xlsx"
    print(f'data_file={data_file}')
    x_ple, y, split_indices, prep = load_and_prepare_data(data_file)

    # 训练
    model, x_tensor, best_epoch, best_val_result = train_model(
        x_ple=x_ple, y=y, split_indices=split_indices, n_classes=len(prep["classes"]))

    # Validation
    val_result, _, _ = evaluate(model, x_tensor, y, split_indices["val"])

    # Test
    test_result, test_pred, test_prob = evaluate(model, x_tensor, y, split_indices["test"])

    # --------------------------------------------------------
    # 打印结果
    # --------------------------------------------------------
    print("FINAL RESULT")
    print("Validation:")

    for k, v in val_result.items():
        print(f"  {k}: {v:.4f} ({v * 100:.2f}%)")

    print("Test:")
    for k, v in test_result.items():
        print(f"  {k}: {v:.4f} ({v * 100:.2f}%)")

    print("Classification Report:")
    print(classification_report(y[split_indices["test"]], test_pred, target_names=list(prep["classes"]), digits=4,
                                zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(y[split_indices["test"]], test_pred, ))

    # --------------------------------------------------------
    # 保存 checkpoint
    # --------------------------------------------------------
    save_checkpoint(model, prep, split_indices, best_epoch, val_result, test_result)

    # --------------------------------------------------------
    # 保存逐样本测试结果
    # --------------------------------------------------------
    save_test_predictions(path=RESULT_OUT, test_idx=split_indices["test"], y=y, pred=test_pred, prob=test_prob,
                          classes=prep["classes"])

    print(f"\n模型已保存：{MODEL_OUT}")
    print(f"测试预测已保存：{RESULT_OUT}")


if __name__ == "__main__":
    main()
