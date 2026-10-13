""" PLE 模型 """

import torch
import torch.nn as nn
import torch.nn.functional as F

# NODE
N_TREES = 128
TREE_DEPTH = 4
TREE_OUT = 4


class PLEEncoder(nn.Module):
    """
    将每个数值特征的 PLE bin 表示映射为可训练 embedding。
    """

    def __init__(self, n_features=47, n_bins=20, emb_dim=8):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(
                n_features,
                n_bins,
                emb_dim,
            ) * 0.02
        )

        self.bias = nn.Parameter(
            torch.zeros(
                n_features,
                emb_dim,
            )
        )

        self.norm = nn.LayerNorm(
            n_features * emb_dim
        )

    def forward(self, x):
        # x: [B, F, Bin]

        z = torch.einsum(
            "nfb,fbd->nfd",
            x,
            self.weight,
        ) + self.bias

        z = z.flatten(1)

        return self.norm(z)


# ============================================================
# 4. NODE 模型
# ============================================================

class SoftObliviousEnsemble(nn.Module):
    """
    可微 Oblivious Tree Ensemble。

    每棵树的每一层：
        1. soft feature selection
        2. learnable threshold
        3. sigmoid soft split

    最后对所有叶节点进行概率加权。
    """

    def __init__(self, in_dim, n_trees=128, depth=4, tree_out=4):
        super().__init__()

        self.depth = depth

        # 每棵树、每层选择特征
        self.feature_logits = nn.Parameter(
            torch.randn(
                n_trees,
                depth,
                in_dim,
            ) * 0.01
        )

        # 可学习阈值
        self.threshold = nn.Parameter(
            torch.zeros(
                n_trees,
                depth,
            )
        )

        # split sharpness
        self.log_scale = nn.Parameter(
            torch.zeros(
                n_trees,
                depth,
            )
        )

        # 每棵树所有叶子的输出
        self.leaf_values = nn.Parameter(
            torch.randn(
                n_trees,
                2 ** depth,
                tree_out,
            ) * 0.02
        )

        # 叶节点二进制路径
        bits = [
            [
                (i >> d) & 1
                for d in range(depth)
            ]
            for i in range(2 ** depth)
        ]

        self.register_buffer(
            "bits",
            torch.tensor(
                bits,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        # soft feature selection
        select = torch.softmax(
            self.feature_logits,
            dim=-1,
        )

        # [B, tree, depth]
        s = torch.einsum(
            "bi,tdi->btd",
            x,
            select,
        )

        # soft decision probability
        p = torch.sigmoid(
            (s - self.threshold)
            * (
                    F.softplus(self.log_scale)
                    + 1.0
            )
        )

        bits = self.bits.view(
            1,
            1,
            -1,
            self.depth,
        )

        # 各叶节点路径概率
        leaf_prob = torch.where(
            bits > 0.5,
            p.unsqueeze(2),
            1.0 - p.unsqueeze(2),
        ).prod(dim=-1)

        # 所有叶节点加权
        out = torch.einsum(
            "btl,tlo->bto",
            leaf_prob,
            self.leaf_values,
        )

        return out.flatten(1)


# ============================================================
# 5. PLE + NODE 完整分类模型
# ============================================================

class PLENODE(nn.Module):

    def __init__(
            self,
            n_features=47,
            n_bins=20,
            emb_dim=8,
            n_classes=7,
    ):
        super().__init__()

        in_dim = n_features * emb_dim

        self.ple = PLEEncoder(
            n_features=n_features,
            n_bins=n_bins,
            emb_dim=emb_dim,
        )

        self.node = SoftObliviousEnsemble(
            in_dim=in_dim,
            n_trees=N_TREES,
            depth=TREE_DEPTH,
            tree_out=TREE_OUT,
        )

        # 与 NODE 并联的 residual MLP
        self.res = nn.Sequential(
            nn.Linear(in_dim, 384),
            nn.GELU(),
            nn.LayerNorm(384),
            nn.Dropout(0.08),

            nn.Linear(384, 256),
            nn.GELU(),
            nn.LayerNorm(256),
        )

        # NODE:
        # 128 trees * tree_out 4 = 512 dims
        # residual = 256 dims
        # total = 768 dims
        self.head = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(
                256,
                n_classes,
            ),
        )

    def forward_features(self, x):
        embedding = self.ple(x)

        node_feature = self.node(
            embedding
        )

        residual_feature = self.res(
            embedding
        )

        return torch.cat(
            [
                node_feature,
                residual_feature,
            ],
            dim=1,
        )

    def forward(self, x):
        feature = self.forward_features(x)

        return self.head(feature)
