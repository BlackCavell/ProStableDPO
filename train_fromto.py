"""From/To 差分头监督 ΔΔG 回归（b 模块，Stability Oracle 式）。

架构：a 模块（ESMtherm + ProteinMPNN + Adapter 掩码训练器）→ 融合表示 za
     → b 模块（From/To 差分头）→ ΔΔG = Linear(h_mt − h_wt)
训练：按蛋白分组，一次前向出该蛋白全部突变，MSE 回归。
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr

from model import SPURSModel
from dataset import MegascaleDataset, ThermoMPNNDataset

ROOT = Path(__file__).resolve().parents[2]
ESMTHERM_CKPT = str(ROOT / "TBD" / "EsmTherm-main" / "output_dir" / "checkpoint-best")
ESM_VANILLA = "facebook/esm2_t33_650M_UR50D"
MPNN_CKPT = str(ROOT / "data" / "proteinmpnn" / "v_48_020.pt")
PDB_DIR = str(ROOT / "data" / "mega_scale_pdb_cache" / "AlphaFold_model_PDBs")
CACHE_DIR = ROOT / "data" / "mpnn_cache"
THERMO_DIR = ROOT / "data" / "megascale" / "thermompnn"


def build_cache(model, dataset, cache_dir, device):
    cache_dir.mkdir(parents=True, exist_ok=True)
    n_new = 0
    model.eval()
    with torch.no_grad():
        for key, feat in dataset.features.items():
            cache_file = cache_dir / f"{key}.pt"
            if cache_file.exists():
                continue
            struct_feats = model.encode_structure(
                feat["X"].unsqueeze(0).to(device), feat["S"].unsqueeze(0).to(device),
                feat["mask"].unsqueeze(0).to(device), feat["residue_idx"].unsqueeze(0).to(device),
                feat["chain"].unsqueeze(0).to(device),
            )[0]
            torch.save(struct_feats.cpu(), cache_file)
            n_new += 1
    print(f"缓存: 新建 {n_new} 个（共 {len(dataset.features)} 蛋白）")
    return n_new


def load_cache(dataset, cache_dir):
    return {key: torch.load(cache_dir / f"{key}.pt") for key in dataset.features}


def group_muts(dataset):
    muts_by_key: dict[str, list] = {}
    for s in dataset.samples:
        muts_by_key.setdefault(s["key"], []).append(s)
    return muts_by_key


@torch.no_grad()
def evaluate(model, test_ds, cache_test, test_muts, device, no_cache=False):
    model.eval()
    all_preds, all_labels = [], []
    per_protein_sp = []
    for key, muts in test_muts.items():
        feat = test_ds.features[key]
        input_ids = feat["input_ids"].unsqueeze(0).to(device)
        chain_idx = torch.tensor([m["chain_idx"] for m in muts], device=device)
        wt_idx = torch.tensor([m["wt_idx"] for m in muts], device=device)
        mt_idx = torch.tensor([m["mt_idx"] for m in muts], device=device)
        if no_cache:
            pred = model.predict_ddg_raw(input_ids,
                feat["X"].unsqueeze(0).to(device), feat["S"].unsqueeze(0).to(device),
                feat["mask"].unsqueeze(0).to(device), feat["residue_idx"].unsqueeze(0).to(device),
                feat["chain"].unsqueeze(0).to(device), chain_idx, wt_idx, mt_idx)
        else:
            pred = model.predict_ddg(input_ids, cache_test[key].unsqueeze(0).to(device),
                                    chain_idx, wt_idx, mt_idx)
        p = pred.cpu().numpy()
        l = np.array([m["ddG"] for m in muts])
        all_preds.append(p); all_labels.append(l)
        if len(p) >= 2 and np.std(l) > 0:
            sp = spearmanr(p, l).correlation
            if not np.isnan(sp):
                per_protein_sp.append(sp)
    all_preds = np.concatenate(all_preds); all_labels = np.concatenate(all_labels)
    pooled_sp = spearmanr(all_preds, all_labels).correlation
    pooled_pc = pearsonr(all_preds, all_labels).statistic
    median_sp = float(np.median(per_protein_sp))
    return pooled_sp, pooled_pc, median_sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default=str(ROOT / "data" / "megascale" / "processed" / "train_ddg.csv"))
    ap.add_argument("--test_csv", default=str(ROOT / "data" / "megascale" / "processed" / "test_ddg.csv"))
    ap.add_argument("--data", default="megascale", choices=["megascale", "thermompnn"],
                    help="megascale=我们的 Dataset1 划分；thermompnn=SPURS 的 MMSeqs 划分")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_proteins", type=int, default=None)
    ap.add_argument("--esm_model", default=ESM_VANILLA, help="ESM 骨干（HF 名或 ESMtherm checkpoint 路径）")
    ap.add_argument("--esm_therm", action="store_true", help="用 ESMtherm 微调权重（否则 vanilla ESM-2）")
    ap.add_argument("--freeze_esm", type=int, default=1)
    ap.add_argument("--freeze_mpnn", type=int, default=1)
    ap.add_argument("--unfreeze_last_n", type=int, default=0, help="解冻 ESM 最后 N 层（>0 时覆盖 freeze_esm）")
    ap.add_argument("--no_cache", action="store_true", help="在线跑 ProteinMPNN encoder（微调 MPNN 时需用，否则梯度到不了）")
    ap.add_argument("--head", default="fromto", choices=["fromto", "l20"], help="预测头：fromto=From/To 差分，l20=SPURS L×20 矩阵头")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    print("加载数据...")
    if args.data == "thermompnn":
        train_ds = ThermoMPNNDataset(str(THERMO_DIR / "mega_train.csv"), PDB_DIR, max_proteins=args.max_proteins)
        test_ds = ThermoMPNNDataset(str(THERMO_DIR / "mega_test.csv"), PDB_DIR, max_proteins=args.max_proteins)
    else:
        train_ds = MegascaleDataset(args.train_csv, PDB_DIR, max_proteins=args.max_proteins)
        test_ds = MegascaleDataset(args.test_csv, PDB_DIR, max_proteins=args.max_proteins)
    print(f"train {train_ds.stats()['n_samples']} 样本/{train_ds.stats()['n_proteins']} 蛋白 | "
          f"test {test_ds.stats()['n_samples']} 样本/{test_ds.stats()['n_proteins']} 蛋白")

    print("初始化模型...")
    model = SPURSModel(args.esm_model, MPNN_CKPT, freeze_esm=bool(args.freeze_esm),
                       freeze_mpnn=bool(args.freeze_mpnn), esm_therm=args.esm_therm,
                       unfreeze_last_n=args.unfreeze_last_n, head=args.head).to(device)
    print(f"trainable {model.trainable_param_count:,} / total {model.total_param_count:,}")

    build_cache(model, train_ds, CACHE_DIR, device)
    build_cache(model, test_ds, CACHE_DIR, device)
    cache_train = load_cache(train_ds, CACHE_DIR)
    cache_test = load_cache(test_ds, CACHE_DIR)
    train_muts = group_muts(train_ds)
    test_muts = group_muts(test_ds)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss, n = 0.0, 0
        t0 = time.time()
        for key, muts in train_muts.items():
            feat = train_ds.features[key]
            input_ids = feat["input_ids"].unsqueeze(0).to(device)
            chain_idx = torch.tensor([m["chain_idx"] for m in muts], device=device)
            wt_idx = torch.tensor([m["wt_idx"] for m in muts], device=device)
            mt_idx = torch.tensor([m["mt_idx"] for m in muts], device=device)
            ddg = torch.tensor([m["ddG"] for m in muts], device=device, dtype=torch.float32)
            if args.no_cache:
                pred = model.predict_ddg_raw(input_ids,
                    feat["X"].unsqueeze(0).to(device), feat["S"].unsqueeze(0).to(device),
                    feat["mask"].unsqueeze(0).to(device), feat["residue_idx"].unsqueeze(0).to(device),
                    feat["chain"].unsqueeze(0).to(device), chain_idx, wt_idx, mt_idx)
            else:
                pred = model.predict_ddg(input_ids, cache_train[key].unsqueeze(0).to(device),
                                        chain_idx, wt_idx, mt_idx)
            loss = F.mse_loss(pred, ddg)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * len(muts); n += len(muts)
        sp, pc, med_sp = evaluate(model, test_ds, cache_test, test_muts, device, no_cache=args.no_cache)
        print(f"epoch {epoch+1}/{args.epochs} | {time.time()-t0:.0f}s | MSE {total_loss/n:.4f} | "
              f"pooled Spearman {sp:.4f} | PCC {pc:.4f} | median per-protein {med_sp:.4f}")


if __name__ == "__main__":
    main()
