"""验证 MegascaleDataset 正确性：加载子集，检查统计与样本映射。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import MegascaleDataset, MPNN_AA_ORDER

CSV = r"F:/Project/Stablity/data/megascale/processed/train_ddg.csv"
PDB_DIR = r"F:/Project/Stablity/data/mega_scale_pdb_cache/AlphaFold_model_PDBs"


def main():
    ds = MegascaleDataset(CSV, PDB_DIR, max_proteins=20)
    st = ds.stats()
    print("== 统计 ==")
    for k, v in st.items():
        print(f"  {k}: {v}")

    print("\n== 前 3 个样本 ==")
    for i in range(3):
        input_ids, X, S, mask, residue_idx, chain, chain_idx, wt_idx, mt_idx, ddG = ds[i]
        print(f"  [{i}] L={S.shape[0]:3d} | input_ids={len(input_ids):3d} | "
              f"pos={chain_idx:3d} {MPNN_AA_ORDER[wt_idx]}->{MPNN_AA_ORDER[mt_idx]} | ddG={ddG:+.3f}")


if __name__ == "__main__":
    main()
