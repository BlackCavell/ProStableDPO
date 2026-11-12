"""结构感知掩码语言模型：MLM 自监督训练 + 零样本稳定性打分。

流程：
  1. 构建结构特征缓存 data/mpnn_cache/{key}.pt（ProteinMPNN 冻结，每蛋白一次）
  2. MLM 训练：Megascale WT 序列 masked AA 预测（自监督，无 ΔΔG 标签）
  3. 零样本打分：log P(mt) − log P(wt) vs 实验 ΔΔG → Spearman/PCC
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr

from model import SPURSModel, ESM_IDS_21AA, MASK_ID, PAD_ID
from dataset import MegascaleDataset

ROOT = Path(__file__).resolve().parents[2]
ESM_CKPT = str(ROOT / "TBD" / "EsmTherm-main" / "output_dir" / "checkpoint-best")
MPNN_CKPT = str(ROOT / "data" / "proteinmpnn" / "v_48_020.pt")
PDB_DIR = str(ROOT / "data" / "mega_scale_pdb_cache" / "AlphaFold_model_PDBs")
CACHE_DIR = ROOT / "data" / "mpnn_cache"


# ---------- 缓存 ----------
def build_cache(model, dataset, cache_dir, device):
    """为每个蛋白预计算 ProteinMPNN 结构特征并缓存（冻结，只需一次）。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    n_new = 0
    t0 = time.time()
    model.eval()
    with torch.no_grad():
        for key, feat in dataset.features.items():
            cache_file = cache_dir / f"{key}.pt"
            if cache_file.exists():
                continue
            X = feat["X"].unsqueeze(0).to(device)
            S = feat["S"].unsqueeze(0).to(device)
            mask = feat["mask"].unsqueeze(0).to(device)
            residue_idx = feat["residue_idx"].unsqueeze(0).to(device)
            chain = feat["chain"].unsqueeze(0).to(device)
            struct_feats = model.encode_structure(X, S, mask, residue_idx, chain)[0]  # [L, 512]
            torch.save(struct_feats.cpu(), cache_file)
            n_new += 1
    print(f"缓存: 新建 {n_new} 个（共 {len(dataset.features)} 蛋白），耗时 {time.time()-t0:.1f}s")
    return n_new


def load_cache(dataset, cache_dir):
    return {key: torch.load(cache_dir / f"{key}.pt") for key in dataset.features}


# ---------- MLM 掩码 ----------
def mask_sequence(input_ids, mask_prob=0.15):
    """随机 mask 15% 残基位（不含 CLS/EOS），返回 (masked_ids, mask_pos[L])。"""
    L = len(input_ids) - 2
    mask_pos = torch.rand(L) < mask_prob
    if mask_pos.sum() == 0:
        mask_pos[0] = True
    masked_ids = input_ids.clone()
    masked_ids[1 : 1 + L][mask_pos] = MASK_ID
    return masked_ids, mask_pos


# ---------- 零样本打分 ----------
@torch.no_grad()
def zero_shot_protein(model, input_ids, struct_feats, device):
    """一次前向 → [L, 21] 的 log P，供该蛋白所有突变复用。"""
    logits = model(input_ids.unsqueeze(0).to(device), struct_feats.unsqueeze(0).to(device))  # [1, L, 33]
    logits_21 = logits[:, :, ESM_IDS_21AA]                                                 # [1, L, 21]
    return F.log_softmax(logits_21, dim=-1)[0]                                             # [L, 21]


def evaluate_zero_shot(model, test_ds, cache, device):
    """按蛋白分组做零样本打分，返回 (scores, labels) 用于算 Spearman/PCC。"""
    # 按蛋白分组突变
    muts_by_key: dict[str, list] = {}
    for s in test_ds.samples:
        muts_by_key.setdefault(s["key"], []).append(s)
    scores, labels = [], []
    model.eval()
    for key, muts in muts_by_key.items():
        feat = test_ds.features[key]
        logp = zero_shot_protein(model, feat["input_ids"], cache[key], device)  # [L, 21]
        for s in muts:
            sc = -(logp[s["chain_idx"], s["mt_idx"]] - logp[s["chain_idx"], s["wt_idx"]])
            scores.append(sc.item()); labels.append(s["ddG"])
    return np.array(scores), np.array(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default=str(ROOT / "data" / "megascale" / "processed" / "train_ddg.csv"))
    ap.add_argument("--test_csv", default=str(ROOT / "data" / "megascale" / "processed" / "test_ddg.csv"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_proteins", type=int, default=None, help="限制蛋白数做快速冒烟")
    ap.add_argument("--rebuild_cache", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    print("加载数据...")
    train_ds = MegascaleDataset(args.train_csv, PDB_DIR, max_proteins=args.max_proteins)
    test_ds = MegascaleDataset(args.test_csv, PDB_DIR, max_proteins=args.max_proteins)
    print(f"train {train_ds.stats()['n_samples']} 样本 / {train_ds.stats()['n_proteins']} 蛋白 | "
          f"test {test_ds.stats()['n_samples']} 样本 / {test_ds.stats()['n_proteins']} 蛋白")

    print("初始化模型...")
    model = SPURSModel(ESM_CKPT, MPNN_CKPT, freeze_esm=False, freeze_mpnn=True).to(device)
    print(f"trainable {model.trainable_param_count:,} / total {model.total_param_count:,}")

    # 缓存结构特征
    build_cache(model, train_ds, CACHE_DIR, device)
    build_cache(model, test_ds, CACHE_DIR, device)
    cache_train = load_cache(train_ds, CACHE_DIR)
    cache_test = load_cache(test_ds, CACHE_DIR)

    # 零样本基线（训练前）
    s0, l0 = evaluate_zero_shot(model, test_ds, cache_test, device)
    print(f"[训练前] 零样本 Spearman {spearmanr(s0, l0).correlation:.4f} | PCC {pearsonr(s0, l0).statistic:.4f}")

    # MLM 训练
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    train_keys = list(train_ds.features.keys())
    for epoch in range(args.epochs):
        model.train()
        total_loss, n = 0.0, 0
        t0 = time.time()
        for key in train_keys:
            feat = train_ds.features[key]
            input_ids = feat["input_ids"]
            struct_feats = cache_train[key]
            masked_ids, mask_pos = mask_sequence(input_ids)
            logits = model(masked_ids.unsqueeze(0).to(device), struct_feats.unsqueeze(0).to(device))  # [1, L, 33]
            targets = input_ids[1 : 1 + len(input_ids) - 2][mask_pos].to(device)
            loss = F.cross_entropy(logits[0][mask_pos], targets)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); n += 1
        s, l = evaluate_zero_shot(model, test_ds, cache_test, device)
        print(f"epoch {epoch+1}/{args.epochs} | {time.time()-t0:.0f}s | MLM loss {total_loss/n:.4f} | "
              f"零样本 Spearman {spearmanr(s, l).correlation:.4f} | PCC {pearsonr(s, l).statistic:.4f}")


if __name__ == "__main__":
    main()
