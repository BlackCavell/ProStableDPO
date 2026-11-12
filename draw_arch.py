"""绘制 v2 模型架构流程图（PNG），输出到 Doc/v2模型架构.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(12, 17))
ax.set_xlim(0, 14)
ax.set_ylim(0, 20)
ax.axis("off")

# 配色
C_INPUT = "#F2F2F2"
C_SEQ = "#D6E4F0"      # 蓝：序列分支
C_STRUCT = "#D9EAD3"   # 绿：结构分支
C_FUSION = "#FCE5CD"   # 橙：融合
C_HEAD = "#F4CCCC"     # 红：预测头
C_OUT = "#EDEDED"      # 灰：输出/损失
EC = "#444444"


def box(x, y, w, h, text, fc, fs=10.5, bold=False, ec=EC):
    b = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.18",
                       linewidth=1.6, edgecolor=ec, facecolor=fc)
    ax.add_patch(b)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.5)


def arrow(x1, y1, x2, y2, color="#555555", ls="-"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                        mutation_scale=16, linewidth=1.6, color=color, linestyle=ls)
    ax.add_patch(a)


# ---- 标题 ----
ax.text(7, 19.4, "v2 模型架构：序列 + 结构融合（SPURS Adapter）+ L×20 头",
        ha="center", va="center", fontsize=14, fontweight="bold")

# ---- 输入 ----
box(7, 17.9, 7.5, 1.05, "输入：WT 序列  +  WT 结构（PDB / AlphaFold）", C_INPUT, bold=True)

# ---- 两个分支 ----
box(3.4, 15.2, 5.3, 1.9, "序列分支  ESMtherm\nESM-2 35M（冻结）\n→ ze  [L × 480]", C_SEQ, bold=True)
box(10.6, 15.2, 5.3, 1.9, "结构分支  ProteinMPNN\nencoder（可微调）\n→ zm  [L × 256]", C_STRUCT, bold=True)

# ---- zs_proj ----
box(10.6, 13.1, 4.4, 1.0, "zs_proj  256 → 480\n→ zs  [L × 480]", C_STRUCT)

# ---- 融合 Adapter ----
box(7, 11.0, 9.2, 1.8, "SPURS Adapter（交叉注意力）\nMHA(ze, zs, zs) + ze  →  W1 → GELU → W2\n→ za  [L × 480]", C_FUSION, bold=True)

# ---- zo ----
box(7, 8.8, 7.8, 1.1, "za_proj(480→128)   ⊕   concat(zm, 256)\n→ zo  [L × 384]", C_FUSION)

# ---- head ----
box(7, 6.7, 5.6, 1.1, "预测头  MLP  384→384→20\n→ phi  [L × 20]", C_HEAD, bold=True)

# ---- ΔΔG ----
box(7, 4.7, 6.8, 1.1, "ΔΔG(i, wt→mt) = phi[i, mt] − phi[i, wt]\n（反称性：self-mutation = 0）", C_OUT)

# ---- loss / eval ----
box(7, 2.7, 6.8, 1.1, "MSE loss → AdamW（只训 4.07M 参数）\n评测：Spearman / PCC", C_OUT)

# ---- 箭头 ----
arrow(7, 17.37, 4.0, 16.15)          # 输入 → 序列分支
arrow(7, 17.37, 10.0, 16.15)         # 输入 → 结构分支
arrow(10.6, 14.25, 10.6, 13.6)       # 结构分支 → zs_proj
arrow(3.4, 14.25, 4.3, 11.9)         # 序列分支 → Adapter
arrow(10.6, 12.6, 9.7, 11.9)         # zs_proj → Adapter
arrow(7, 10.1, 7, 9.35)              # Adapter → zo
arrow(7, 8.25, 7, 7.25)              # zo → head
arrow(7, 6.15, 7, 5.25)              # head → ΔΔG
arrow(7, 4.15, 7, 3.25)              # ΔΔG → loss

# ---- 图例 ----
lg = [("输入", C_INPUT), ("序列分支（冻结）", C_SEQ), ("结构分支（可微调）", C_STRUCT),
      ("融合", C_FUSION), ("预测头", C_HEAD)]
x0 = 1.6
for i, (label, color) in enumerate(lg):
    x = x0 + i * 2.4
    ax.add_patch(FancyBboxPatch((x - 0.55, 0.7), 1.1, 0.5,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                linewidth=1.0, edgecolor=EC, facecolor=color))
    ax.text(x, 0.95, label, ha="center", va="center", fontsize=8.5)

out = r"F:/Project/Stablity/Doc/v2模型架构.png"
plt.tight_layout()
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
print("saved:", out)
