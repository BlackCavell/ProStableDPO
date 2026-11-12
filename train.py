"""阶段 0 训练：Megascale ΔΔG，MSE 回归，Spearman/PCC 评测。"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from scipy.stats import spearmanr, pearsonr

from model import SPURSModel
from dataset import MegascaleDataset, PAD

ROOT = Path(__file__).resolve().parents[2]
ESM_CKPT = str(ROOT / "TBD" / "EsmTherm-main" / "output_dir" / "checkpoint-best")
MPNN_CKPT = str(ROOT / "data" / "proteinmpnn" / "v_48_020.pt")
PDB_DIR = str(ROOT / "data" / "mega_scale_pdb_cache" / "AlphaFold_model_PDBs")


def collate(batch):
    """变长序列 padding 成一个 batch。"""
    input_ids_l, X_l, S_l, mask_l, ridx_l, chain_l = [], [], [], [], [], []
    cidx_l, wt_l, mt_l, ddG_l = [], [], [], []
    for (input_ids, X, S, mask, residue_idx, chain, cidx, wt, mt, ddG) in batch:
        input_ids_l.append(input_ids); X_l.append(X); S_l.append(S)
        mask_l.append(mask); ridx_l.append(residue_idx); chain_l.append(chain)
        cidx_l.append(cidx); wt_l.append(wt); mt_l.append(mt); ddG_l.append(ddG)

    B = len(batch)
    L_max = max(s.shape[0] for s in S_l)
    input_ids = pad_sequence(input_ids_l, batch_first=True, padding_value=PAD)  # [B, L_max+2]
    X = torch.zeros(B, L_max, 4, 3)
    S = torch.zeros(B, L_max, dtype=torch.long)
    mask = torch.zeros(B, L_max)
    residue_idx = torch.full((B, L_max), -100, dtype=torch.long)
    chain = torch.zeros(B, L_max, dtype=torch.long)
    for i in range(B):
        L = S_l[i].shape[0]
        X[i, :L] = X_l[i]; S[i, :L] = S_l[i]; mask[i, :L] = mask_l[i]
        residue_idx[i, :L] = ridx_l[i]; chain[i, :L] = chain_l[i]

    cidx = torch.tensor(cidx_l, dtype=torch.long)
    wt = torch.tensor(wt_l, dtype=torch.long)
    mt = torch.tensor(mt_l, dtype=torch.long)
    ddG = torch.tensor(ddG_l, dtype=torch.float32)
    return input_ids, X, S, mask, residue_idx, chain, cidx, wt, mt, ddG


def predict_ddG(model, input_ids, X, S, mask, residue_idx, chain, cidx, wt, mt):
    phi = model(input_ids, X, S, mask, residue_idx, chain)   # [B, L, 20]
    b = torch.arange(phi.shape[0], device=phi.device)
    return phi[b, cidx, mt] - phi[b, cidx, wt]               # [B]


def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for (input_ids, X, S, mask, residue_idx, chain, cidx, wt, mt, ddG) in loader:
            input_ids, X, S, mask = (t.to(device) for t in (input_ids, X, S, mask))
            residue_idx, chain, cidx, wt, mt = (t.to(device) for t in (residue_idx, chain, cidx, wt, mt))
            p = predict_ddG(model, input_ids, X, S, mask, residue_idx, chain, cidx, wt, mt)
            preds.append(p.cpu().numpy()); labels.append(ddG.numpy())
    preds = np.concatenate(preds); labels = np.concatenate(labels)
    return spearmanr(preds, labels).correlation, pearsonr(preds, labels).statistic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default=str(ROOT / "data" / "megascale" / "processed" / "train_ddg.csv"))
    ap.add_argument("--test_csv", default=str(ROOT / "data" / "megascale" / "processed" / "test_ddg.csv"))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_proteins", type=int, default=None, help="限制蛋白数做快速冒烟")
    ap.add_argument("--freeze_esm", type=int, default=0)
    ap.add_argument("--freeze_mpnn", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    print("加载数据...")
    train_ds = MegascaleDataset(args.train_csv, PDB_DIR, max_proteins=args.max_proteins)
    test_ds = MegascaleDataset(args.test_csv, PDB_DIR, max_proteins=args.max_proteins)
    print("train stats:", train_ds.stats())
    print("test  stats:", test_ds.stats())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    print("初始化模型...")
    model = SPURSModel(ESM_CKPT, MPNN_CKPT, freeze_esm=bool(args.freeze_esm),
                       freeze_mpnn=bool(args.freeze_mpnn)).to(device)
    print(f"trainable params: {model.trainable_param_count:,} / {model.total_param_count:,}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss, n = 0.0, 0
        t0 = time.time()
        for (input_ids, X, S, mask, residue_idx, chain, cidx, wt, mt, ddG) in train_loader:
            input_ids, X, S, mask = (t.to(device) for t in (input_ids, X, S, mask))
            residue_idx, chain, cidx, wt, mt, ddG = (t.to(device) for t in (residue_idx, chain, cidx, wt, mt, ddG))
            pred = predict_ddG(model, input_ids, X, S, mask, residue_idx, chain, cidx, wt, mt)
            loss = F.mse_loss(pred, ddG)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * len(ddG); n += len(ddG)
        train_t = time.time() - t0
        sp, pc = evaluate(model, test_loader, device)
        gpu_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        print(f"epoch {epoch+1}/{args.epochs} | 耗时 {train_t:.0f}s | 吞吐 {n/train_t:.0f}样本/s | "
              f"train MSE {total_loss/n:.4f} | test Spearman {sp:.4f} | PCC {pc:.4f} | GPU峰值 {gpu_gb:.2f}GB")


if __name__ == "__main__":
    main()
