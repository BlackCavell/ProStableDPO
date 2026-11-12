"""阶段 0 smoke test：验证 ESMtherm + ProteinMPNN 两个 checkpoint 能加载、前向能跑通。"""
import sys, os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import SPURSModel

ESM_CKPT = r"F:/Project/Stablity/TBD/EsmTherm-main/output_dir/checkpoint-best"
MPNN_CKPT = r"F:/Project/Stablity/data/proteinmpnn/v_48_020.pt"


def main():
    print("== 实例化 SPURSModel（加载两个 checkpoint）==")
    model = SPURSModel(esmtherm_ckpt=ESM_CKPT, mpnn_ckpt=MPNN_CKPT, freeze_esm=True)
    print(f"  total params    : {model.total_param_count:,}")
    print(f"  trainable params: {model.trainable_param_count:,}")
    print(f"  frozen params   : {model.total_param_count - model.trainable_param_count:,}")

    print("\n== 合成前向（随机输入，仅验证 shape 与连通）==")
    L = 12
    # ESM 输入：CLS(0) + 序列 + EOS(2)
    seq_tokens = torch.randint(4, 24, (1, L))          # 20 种标准 AA 的 token id
    input_ids = torch.cat([
        torch.zeros(1, 1, dtype=torch.long),
        seq_tokens,
        torch.full((1, 1), 2, dtype=torch.long),
    ], dim=1)                                           # [1, L+2]

    X = torch.randn(1, L, 4, 3)                         # N/Ca/C/O 坐标
    S = torch.randint(0, 20, (1, L))                    # AA 索引 0..19
    mask = torch.ones(1, L)
    residue_idx = torch.arange(L).unsqueeze(0)
    chain_encoding_all = torch.zeros(1, L, dtype=torch.long)

    model.eval()
    with torch.no_grad():
        phi = model(input_ids, X, S, mask, residue_idx, chain_encoding_all)

    print(f"  phi shape: {tuple(phi.shape)}  (期望 [1, {L}, 20])")
    assert tuple(phi.shape) == (1, L, 20), f"shape 错误: {tuple(phi.shape)}"

    # 验证 From/To 差分（反称性：self-mutation = 0）
    wt_idx = torch.tensor([[5] * L])
    mt_idx = torch.tensor([[5] * L])                    # 同 index → ΔΔG 应为 0
    ddG_self = model.predict_ddG(phi, wt_idx, mt_idx)
    print(f"  self-mutation ΔΔG (应≈0): max |ddG| = {ddG_self.abs().max().item():.2e}")

    print("\n[PASS] smoke test OK")


if __name__ == "__main__":
    main()
