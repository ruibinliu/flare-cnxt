"""Prepare data."""
import os
import shutil

import pandas as pd
from nvflare.app_opt.feature_election import FeatureElection

from config import Config


def split_and_save(
        excel_path="./FV线性全.xlsx",
        output_dir="./federated_data",
        target_col=None,  # 标签列名；若为 None 则用第一列
        num_clients=5,
        split_strategy="stratified",  # "stratified" 或 "non_iid"
        dirichlet_alpha=0.5,  # 仅 non_iid 时生效
        random_state=42):
    # ---------------------------------------------------------
    # 1. 读入完整数据（保持原结构，不做任何列筛选）
    # ---------------------------------------------------------
    df = pd.read_excel(excel_path)
    print(f"读取数据：{excel_path}, shape={df.shape}")

    if target_col is None:
        target_col = df.columns[0]  # A列作为标签
    print(f"标签列：{target_col}")

    # ---------------------------------------------------------
    # 2. 用 prepare_data_splits 得到每个客户端的划分
    #    注意：这里喂的是完整 df（含标签列），API 会自动把
    #    target_col 拆出来作为 y
    # ---------------------------------------------------------
    fe = FeatureElection(freedom_degree=0.5, fs_method="lasso")

    client_splits = fe.prepare_data_splits(
        df=df,
        target_col=target_col,
        num_clients=num_clients,
        split_strategy=split_strategy,
        random_state=random_state,
        dirichlet_alpha=dirichlet_alpha,
    )

    print(f"\n划分完成：num_clients={num_clients}, strategy={split_strategy}, alpha={dirichlet_alpha}")

    # ---------------------------------------------------------
    # 3. 从每个 X_client 提取行索引，切原始完整 df
    # ---------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    for cid, (X_client, y_client) in enumerate(client_splits):
        site_name = f"site-{cid + 1}"

        # ★ 关键：X_client.index 就是该客户端在原始 df 中的行索引
        row_idx = X_client.index.to_numpy()

        # 用索引切原始 df —— 结构完全不变
        df_client = df.iloc[row_idx].reset_index(drop=True)

        # 目录
        site_dir = os.path.join(output_dir, site_name)
        os.makedirs(site_dir, exist_ok=True)

        # 保存
        out_path = os.path.join(site_dir, "data.xlsx")
        df_client.to_excel(out_path, index=False)

        # 打印该客户端标签分布
        label_cnt = df_client[target_col].value_counts().to_dict()
        print(f"  {site_name}: n={len(df_client):5d}  "
              f"标签分布={label_cnt}  → {out_path}")

    print(f"\n全部完成，输出目录：{os.path.abspath(output_dir)}")


def main():
    shutil.copytree(src="data", dst=Config.RUNTIME_DATA_ROOT, dirs_exist_ok=True)
    print(f"Synced dataset to runtime directory")


if __name__ == "__main__":
    split_and_save(
        excel_path="./data/raw/pulmonary.xlsx",
        output_dir="./data",
        target_col="肺功能常规类型",  # None = 用第一列作标签
        num_clients=5,
        split_strategy="stratified",  # 先跑 IID baseline
        # split_strategy="non_iid",    # 想跑 Non-IID 时切换
        # dirichlet_alpha=0.5,
        random_state=42
    )

    main()
