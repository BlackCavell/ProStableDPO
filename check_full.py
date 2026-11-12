"""全量数据加载检查：加载全部 train/test，报告统计、耗时、内存，暴露规模性问题。"""
import sys, os, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import MegascaleDataset

CSV_TRAIN = r"F:/Project/Stablity/data/megascale/processed/train_ddg.csv"
CSV_TEST = r"F:/Project/Stablity/data/megascale/processed/test_ddg.csv"
PDB_DIR = r"F:/Project/Stablity/data/mega_scale_pdb_cache/AlphaFold_model_PDBs"


def load(name, csv):
    t = time.time()
    ds = MegascaleDataset(csv, PDB_DIR)
    dt = time.time() - t
    st = ds.stats()
    print(f"[{name}] 耗时 {dt:.1f}s | 样本 {st['n_samples']} | 蛋白 {st['n_proteins']} "
          f"| 多点 {st['n_multi']} | 无结构 {st['n_no_struct']} | 映射失败 {st['n_bad_map']} "
          f"| 解析失败 {st['n_parse_err']}")
    return ds


def main():
    t0 = time.time()
    train = load("train", CSV_TRAIN)
    test = load("test", CSV_TEST)
    print(f"\n总耗时 {time.time()-t0:.1f}s | train {len(train)} 样本 | test {len(test)} 样本")
    if torch.cuda.is_available():
        print(f"GPU 显存 allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
