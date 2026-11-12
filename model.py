"""SPURS 式结构感知掩码语言模型（v2，重构版）。

对齐 SPURS 源码（TBD/SPURS-main）：
- 序列分支：ESMtherm（ESM-2 35M，可微调）。
- 结构分支：ProteinMPNN encoder（冻结），结构特征 = concat([h_V0, W_s(S), h_V1, h_V2]) [L, 512]。
- Adapter：交叉注意力 + bottleneck FFN（Pre-LN 残差），融合结构进序列。
- 掩码头：lm_head（Linear 480→33），做 masked AA 预测（自监督）。
- 稳定性预测：零样本打分 log P(mt) − log P(wt)（无需回归头、无 ΔΔG 标签）。
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EsmForSequenceClassification, EsmModel

_MPNN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "proteinmpnn")
if _MPNN_DIR not in sys.path:
    sys.path.insert(0, _MPNN_DIR)
from protein_mpnn_utils import ProteinMPNN, gather_nodes  # noqa: E402

# ---- 常量 ----
ESM_HIDDEN = 480        # esm2_t12_35M_UR50D 隐藏维
ESM_VOCAB = 33          # ESM-2 词表大小
ESM_HEADS = 20          # esm2_t12_35M 注意力头数
MPNN_HIDDEN = 128       # ProteinMPNN 隐藏维
MPNN_VOCAB = 21         # 20 AA + 1 特殊
STRUCT_FEAT_DIM = 4 * MPNN_HIDDEN   # concat([h_V0, W_s, h_V1, h_V2]) = 512
PAD_ID = 1
MASK_ID = 32

# ESM-2 词表顺序 'LAGVSERTIDPKQNFYMHWC'（id 4..23）+ 'X'(24)
# ProteinMPNN alphabet 'ACDEFGHIKLMNPQRSTVWYX'（0..20）
# 21 种 AA（ProteinMPNN 顺序）对应的 ESM token id
MPNN_AA_ORDER = "ACDEFGHIKLMNPQRSTVWYX"
ESM_AA_TO_ID = {aa: 4 + i for i, aa in enumerate("LAGVSERTIDPKQNFYMHWC")}
ESM_AA_TO_ID["X"] = 24
ESM_IDS_21AA = [ESM_AA_TO_ID[aa] for aa in MPNN_AA_ORDER]   # 21 个 ESM token id，按 MPNN 顺序


class ProteinMPNNStructureEncoder(nn.Module):
    """ProteinMPNN encoder，返回 SPURS 风格结构特征 concat([h_V0, W_s, h_V1, h_V2]) [L, 512]。"""

    def __init__(self, mpnn: ProteinMPNN):
        super().__init__()
        self.features = mpnn.features
        self.W_e = mpnn.W_e
        self.W_s = mpnn.W_s
        self.encoder_layers = mpnn.encoder_layers

    def forward(self, X, S, mask, residue_idx, chain_encoding_all):
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        h_V_list = []
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
            h_V_list.append(h_V)
        h_S = self.W_s(S)
        # SPURS: torch.cat([all_mpnn_hid[0], mpnn_embed, all_mpnn_hid[1], all_mpnn_hid[2]], -1)
        return torch.cat([h_V_list[0], h_S, h_V_list[1], h_V_list[2]], dim=-1)  # [B, L, 512]


class SPURSAdapter(nn.Module):
    """SPURS 结构适配器：交叉注意力 + bottleneck FFN，均 Pre-LN 残差。

    对应源码 structural_adapter_attn（NormalizedResidualBlock 包 MultiheadAttention）
    与 structural_adapter_ffn（NormalizedResidualBlock 包 FeedForwardNetwork，bottleneck embed/2）。
    """

    def __init__(self, dim=ESM_HIDDEN, structure_dim=STRUCT_FEAT_DIM, num_heads=ESM_HEADS, dropout=0.1):
        super().__init__()
        self.zs_proj = nn.Linear(structure_dim, dim)          # 512 → 480
        self.attn_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, dim))
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, ze, zs):
        zs = self.zs_proj(zs)                                 # [B, L, 480]
        attn_out, _ = self.cross_attn(self.attn_norm(ze), zs, zs, need_weights=False)
        x = ze + self.attn_dropout(attn_out)                  # 交叉注意力（Pre-LN 残差）
        x = x + self.ffn_dropout(self.ffn(self.ffn_norm(x)))  # bottleneck FFN（Pre-LN 残差）
        return x


class FromToHead(nn.Module):
    """E1 Stability Oracle 式 From/To 差分头（b 模块）。

    用突变位点的融合表示 site + From/To 氨基酸嵌入做 Siamese：
    h = Context(site, e)，ΔΔG = Linear(h_mt − h_wt)（反称性：self-mutation=0）。
    """

    def __init__(self, dim=ESM_HIDDEN, num_aa=20):
        super().__init__()
        self.aa_embed = nn.Embedding(num_aa, dim)                    # From/To AA 嵌入（可换成 lm_head 权重）
        self.context = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.out = nn.Linear(dim, 1)

    def forward(self, site, wt_idx, mt_idx):
        e_wt = self.aa_embed(wt_idx)                                 # [B, 480]
        e_mt = self.aa_embed(mt_idx)
        h_wt = self.context(torch.cat([site, e_wt], dim=-1))         # [B, 480]
        h_mt = self.context(torch.cat([site, e_mt], dim=-1))
        return self.out(h_mt - h_wt).squeeze(-1)                     # [B] ΔΔG


class L20Head(nn.Module):
    """SPURS L×20 矩阵头：phi = MLP(za)，ΔΔG = phi[i,mt] − phi[i,wt]。"""

    def __init__(self, dim=ESM_HIDDEN, num_aa=20):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, num_aa))

    def forward(self, za):
        return self.mlp(za)                                             # [B, L, 20]


class SPURSModel(nn.Module):
    """ESMtherm + ProteinMPNN(冻结) + Adapter + lm_head，结构感知掩码语言模型。"""

    def __init__(self, esm_name_or_path: str, mpnn_ckpt: str, freeze_esm: bool = False,
                 freeze_mpnn: bool = True, esm_therm: bool = False, unfreeze_last_n: int = 0,
                 head: str = "fromto"):
        super().__init__()
        # 序列骨干（vanilla ESM-2 或 ESMtherm，可微调）
        if esm_therm:
            self.esm = EsmForSequenceClassification.from_pretrained(esm_name_or_path).esm
        else:
            self.esm = EsmModel.from_pretrained(esm_name_or_path)
        hidden = self.esm.config.hidden_size
        heads = self.esm.config.num_attention_heads
        if unfreeze_last_n > 0:
            # 冻结全部，仅解冻最后 N 层（参数高效微调）
            for p in self.esm.parameters():
                p.requires_grad = False
            for layer in self.esm.encoder.layer[-unfreeze_last_n:]:
                for p in layer.parameters():
                    p.requires_grad = True
        elif freeze_esm:
            for p in self.esm.parameters():
                p.requires_grad = False
        # 结构编码（默认冻结）
        mpnn = ProteinMPNN(
            num_letters=MPNN_VOCAB, node_features=MPNN_HIDDEN, edge_features=MPNN_HIDDEN,
            hidden_dim=MPNN_HIDDEN, num_encoder_layers=3, num_decoder_layers=3,
            vocab=MPNN_VOCAB, k_neighbors=48, dropout=0.1, ca_only=False,
        )
        state = torch.load(mpnn_ckpt, map_location="cpu")
        mpnn.load_state_dict(state["model_state_dict"])
        self.mpnn_encoder = ProteinMPNNStructureEncoder(mpnn)
        if freeze_mpnn:
            for p in self.mpnn_encoder.parameters():
                p.requires_grad = False
        # Adapter + 掩码头 + 预测头（维度随骨干自适应）
        self.adapter = SPURSAdapter(dim=hidden, num_heads=heads)
        self.lm_head = nn.Linear(hidden, ESM_VOCAB, bias=False)
        self.fromto_head = FromToHead(dim=hidden)
        self.l20_head = L20Head(dim=hidden)
        self.head_mode = head

    def encode_structure(self, X, S, mask, residue_idx, chain_encoding_all):
        """结构特征（ProteinMPNN encoder 输出，可缓存）。"""
        return self.mpnn_encoder(X, S, mask, residue_idx, chain_encoding_all)  # [B, L, 512]

    def _forward_za(self, input_ids, structure_feats):
        """ESMtherm → adapter 融合表示 za [B, L, 480]。"""
        attention_mask = (input_ids != PAD_ID).float()
        ze_full = self.esm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # [B, L+2, 480]
        ze = ze_full[:, 1:-1, :]                                            # 去 CLS/EOS → [B, L, 480]
        return self.adapter(ze, structure_feats)                            # [B, L, 480]

    def forward(self, input_ids, structure_feats):
        """返回掩码头 logits [B, L, 33]（a 模块：自监督 MLM）。"""
        za = self._forward_za(input_ids, structure_feats)
        return self.lm_head(za)                                             # [B, L, 33]

    def predict_ddg(self, input_ids, structure_feats, chain_idx, wt_idx, mt_idx):
        """预测 ΔΔG（按 head_mode 分发：fromto 或 l20）。"""
        za = self._forward_za(input_ids, structure_feats)                   # [B, L, hidden]
        b = torch.arange(za.shape[0], device=za.device)
        if self.head_mode == "l20":
            phi = self.l20_head(za)                                          # [B, L, 20]
            return phi[b, chain_idx, mt_idx] - phi[b, chain_idx, wt_idx]     # [B]
        site = za[b, chain_idx]
        return self.fromto_head(site, wt_idx, mt_idx)                       # [B]

    def predict_ddg_raw(self, input_ids, X, S, mask, residue_idx, chain, chain_idx, wt_idx, mt_idx):
        """预测 ΔΔG，结构特征在线计算（含梯度，用于微调 ProteinMPNN）。"""
        structure_feats = self.mpnn_encoder(X, S, mask, residue_idx, chain)  # [B, L, 512]（有梯度）
        za = self._forward_za(input_ids, structure_feats)
        b = torch.arange(za.shape[0], device=za.device)
        if self.head_mode == "l20":
            phi = self.l20_head(za)
            return phi[b, chain_idx, mt_idx] - phi[b, chain_idx, wt_idx]
        site = za[b, chain_idx]
        return self.fromto_head(site, wt_idx, mt_idx)                       # [B]

    @torch.no_grad()
    def zero_shot_score(self, input_ids, structure_feats, chain_idx, wt_mpnn_idx, mt_mpnn_idx):
        """零样本打分：log P(mt) − log P(wt)，再取负（负值=更失稳）。"""
        logits = self.forward(input_ids, structure_feats)                   # [B, L, 33]
        logits_21 = logits[:, :, ESM_IDS_21AA]                              # [B, L, 21] 按 MPNN 顺序
        logp = F.log_softmax(logits_21, dim=-1)                             # [B, L, 21]
        b = torch.arange(logp.shape[0], device=logp.device)
        score = logp[b, chain_idx, mt_mpnn_idx] - logp[b, chain_idx, wt_mpnn_idx]
        return -score

    @property
    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
