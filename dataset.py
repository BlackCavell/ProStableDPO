"""数据集：Megascale ddG 数据 → 模型输入（ESM input_ids + ProteinMPNN 结构特征 + 突变标签）。

关键设计：
- **以 PDB 链序列作为模型的 WT 序列**（去掉 N/C 端 linker，使 ESM 与 ProteinMPNN 长度一致）。
- 突变经 id 里的 **PDB 残基编号** 映射到链内索引（res_id → 链内 idx）。
- ESM-2 与 ProteinMPNN 的氨基酸顺序不同，映射分别处理：
    ESM-2    vocab id 4..23  = 'LAGVSERTIDPKQNFYMHWC'
    ProteinMPNN alphabet 0..19 = 'ACDEFGHIKLMNPQRSTVWY'（同时是 L×20 头的 AA 顺序）
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Optional

import pandas as pd

import numpy as np
import torch
from torch.utils.data import Dataset
import biotite.structure as struc
import biotite.structure.io as strucio
from biotite.structure import filter_amino_acids

# ---- 氨基酸映射（两个模型顺序不同）----
ESM_AA_ORDER = "LAGVSERTIDPKQNFYMHWC"     # ESM-2 token id 4..23
MPNN_AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"    # ProteinMPNN alphabet 0..19（= L×20 头顺序）
ESM_AA_TO_ID = {aa: 4 + i for i, aa in enumerate(ESM_AA_ORDER)}
MPNN_AA_TO_IDX = {aa: i for i, aa in enumerate(MPNN_AA_ORDER)}
CLS, PAD, EOS = 0, 1, 2

AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}

_SINGLE_MUT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def tokenize_esm(seq: str) -> list[int]:
    """ESM-2 输入：CLS + 序列 + EOS。"""
    return [CLS] + [ESM_AA_TO_ID[aa] for aa in seq] + [EOS]


def parse_pdb(path: str):
    """解析 PDB → (res_ids, seq, X)。

    只保留 N/CA/C/O 四原子齐全的残基。返回：
      res_ids: list[int]      PDB 残基编号（保持顺序）
      seq:     str            链序列（1 字母）
      X:       np.ndarray     [L, 4, 3] 骨干坐标（N,CA,C,O）
    """
    arr = strucio.load_structure(path)
    arr = arr[filter_amino_acids(arr)]
    # 只取第一条链（Megascale mini-protein 多为单链；多链时取首链避免 res_id 跨链冲突）
    arr = arr[arr.chain_id == arr.chain_id[0]]
    res_ids, seq, coords = [], [], []
    for rid in np.unique(arr.res_id):
        res = arr[arr.res_id == rid]
        atoms = {}
        for a in ("N", "CA", "C", "O"):
            m = res[res.atom_name == a]
            if len(m) == 1:
                atoms[a] = m.coord[0]
        if len(atoms) == 4:                      # 完整骨干
            res_ids.append(int(rid))
            seq.append(AA3TO1.get(res.res_name[0], "X"))
            coords.append([atoms["N"], atoms["CA"], atoms["C"], atoms["O"]])
    return res_ids, "".join(seq), np.array(coords, dtype=np.float32)


def build_protein(pdb_path: str) -> dict:
    """为一个蛋白构建模型输入特征（ESM + ProteinMPNN），复用一次前向。"""
    res_ids, chain_seq, X = parse_pdb(pdb_path)
    L = len(chain_seq)
    S = np.array([MPNN_AA_TO_IDX.get(aa, 0) for aa in chain_seq], dtype=np.int64)
    return {
        "input_ids": torch.tensor(tokenize_esm(chain_seq), dtype=torch.long),  # [L+2]
        "X": torch.tensor(X, dtype=torch.float32),                              # [L,4,3]
        "S": torch.tensor(S, dtype=torch.long),                                 # [L]
        "mask": torch.ones(L, dtype=torch.float32),                             # [L]
        "residue_idx": torch.arange(L, dtype=torch.long),                       # [L]
        "chain": torch.zeros(L, dtype=torch.long),                              # [L] 单链
        "chain_seq": chain_seq,
        "res_id_to_idx": {rid: i for i, rid in enumerate(res_ids)},
    }


class MegascaleDataset(Dataset):
    """单点突变 ΔΔG 数据集。

    样本：每个单点突变一条 (input_ids, X, S, mask, residue_idx, chain, wt_idx, mt_idx, ddG)。
    """

    def __init__(self, csv_path: str, pdb_dir: str, max_proteins: Optional[int] = None):
        self.pdb_dir = Path(pdb_dir)
        self.features: dict[str, dict] = {}      # pdb_key -> 蛋白特征
        self.samples: list[dict] = []

        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        n_multi = n_no_struct = n_bad_map = n_parse_err = 0
        seen_proteins: set[str] = set()
        for row in rows:
            pdb = row["pdb_name"]
            key = pdb[:-4] if pdb.endswith(".pdb") else pdb
            # 单点突变过滤：id 的突变部分匹配 ^A#B$
            mut_str = row["id"].split("_")[-1]
            m = _SINGLE_MUT_RE.match(mut_str)
            if not m:
                n_multi += 1
                continue
            wt_aa, position, mt_aa = m.group(1), int(m.group(2)), m.group(3)
            if wt_aa not in MPNN_AA_TO_IDX or mt_aa not in MPNN_AA_TO_IDX:
                continue

            # 蛋白特征（缓存）
            if key not in self.features:
                if max_proteins is not None and len(seen_proteins) >= max_proteins:
                    continue
                pdb_file = self.pdb_dir / f"{key}.pdb"
                if not pdb_file.exists():
                    n_no_struct += 1
                    continue
                try:
                    feat = build_protein(str(pdb_file))
                except Exception:
                    n_parse_err += 1
                    continue
                self.features[key] = feat
                seen_proteins.add(key)

            feat = self.features[key]
            chain_idx = feat["res_id_to_idx"].get(position)
            if chain_idx is None:
                n_bad_map += 1
                continue
            if feat["chain_seq"][chain_idx] != wt_aa:
                n_bad_map += 1                 # 映射不一致，跳过
                continue

            self.samples.append({
                "key": key,
                "chain_idx": chain_idx,
                "wt_idx": MPNN_AA_TO_IDX[wt_aa],
                "mt_idx": MPNN_AA_TO_IDX[mt_aa],
                "ddG": float(row["ddG"]),
            })

        self._stats = {
            "n_rows": len(rows), "n_multi": n_multi, "n_no_struct": n_no_struct,
            "n_bad_map": n_bad_map, "n_parse_err": n_parse_err,
            "n_samples": len(self.samples), "n_proteins": len(self.features),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        f = self.features[s["key"]]
        return (
            f["input_ids"], f["X"], f["S"], f["mask"],
            f["residue_idx"], f["chain"],
            s["chain_idx"], s["wt_idx"], s["mt_idx"], s["ddG"],
        )

    def stats(self) -> dict:
        return dict(self._stats)


class ThermoMPNNDataset(Dataset):
    """ThermoMPNN/SPURS 划分的 Megascale ΔΔG 数据集（Dataset2/3 的 ddG_ML 标签）。

    与 MegascaleDataset 的区别：
    - 数据源：ThermoMPNN 的 mega_train/val/test.csv（MMSeqs 25% 序列聚类划分）
    - 标签：ddG = -ddG_ML（SPURS 约定，取负）
    - 位置：mut_type[1:-1] - 1，直接是 aa_seq（核心域）的 0-index，无 linker offset
    """

    def __init__(self, csv_path, pdb_dir, max_proteins=None):
        self.pdb_dir = Path(pdb_dir)
        self.features: dict[str, dict] = {}
        self.samples: list[dict] = []
        df = pd.read_csv(csv_path)
        n_non_single = n_no_struct = n_parse_err = n_bad_map = 0
        seen: set[str] = set()
        for wt_name, group in df.groupby("WT_name"):
            if max_proteins is not None and len(seen) >= max_proteins:
                break
            wt_rows = group[group["mut_type"] == "wt"]
            if len(wt_rows) == 0:
                continue
            wt_seq = wt_rows["aa_seq"].iloc[0]
            pdb_file = self.pdb_dir / wt_name
            if not pdb_file.exists():
                n_no_struct += 1
                continue
            try:
                feat = build_protein(str(pdb_file))
            except Exception:
                n_parse_err += 1
                continue
            chain_seq = feat["chain_seq"]
            # 对齐 aa_seq（核心域）与 PDB 链序列
            if chain_seq == wt_seq:
                off = 0
            elif chain_seq in wt_seq:
                off = wt_seq.index(chain_seq)
            elif wt_seq in chain_seq:
                off = -chain_seq.index(wt_seq)
            else:
                continue
            key = wt_name[:-4] if wt_name.endswith(".pdb") else wt_name
            self.features[key] = feat
            seen.add(wt_name)
            for _, row in group[group["mut_type"] != "wt"].iterrows():
                mt = row["mut_type"]
                if ":" in mt or "ins" in mt or "del" in mt:
                    n_non_single += 1
                    continue
                wt_aa, pos, mt_aa = mt[0], int(mt[1:-1]) - 1, mt[-1]
                chain_idx = pos - off
                if not (0 <= chain_idx < len(chain_seq)) or chain_seq[chain_idx] != wt_aa:
                    n_bad_map += 1
                    continue
                if wt_aa not in MPNN_AA_TO_IDX or mt_aa not in MPNN_AA_TO_IDX:
                    continue
                self.samples.append({
                    "key": key, "chain_idx": chain_idx,
                    "wt_idx": MPNN_AA_TO_IDX[wt_aa], "mt_idx": MPNN_AA_TO_IDX[mt_aa],
                    "ddG": -float(row["ddG_ML"]),
                })
        self._stats = {
            "n_samples": len(self.samples), "n_proteins": len(self.features),
            "n_non_single": n_non_single, "n_no_struct": n_no_struct,
            "n_parse_err": n_parse_err, "n_bad_map": n_bad_map,
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        f = self.features[s["key"]]
        return (f["input_ids"], f["X"], f["S"], f["mask"], f["residue_idx"], f["chain"],
                s["chain_idx"], s["wt_idx"], s["mt_idx"], s["ddG"])

    def stats(self) -> dict:
        return dict(self._stats)
