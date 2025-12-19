import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score


# =========================
# 1) 路径：直接改这里
# =========================
CSV_PATH = "/workspace/generation_result/AI-AtomGen-prop-dft_3d-test-rmse_quarter90_qwen2.5.csv"
FIG_PATH = "/workspace/fig_lattice_volume_scatter_Qwen180.png"


# =========================
# 2) POSCAR -> (a,b,c,V)
# =========================
def abcV_from_poscar_str(poscar_str: str):
    """
    返回 (a,b,c,volume)
    解析失败返回 None
    """
    if not isinstance(poscar_str, str):
        return None

    s = poscar_str.replace("\\n", "\n").strip()
    lines = [l.strip() for l in s.splitlines() if l.strip()]
    if len(lines) < 5:
        return None

    try:
        scale = float(lines[1])
        a_vec = np.array(list(map(float, lines[2].split()))) * scale
        b_vec = np.array(list(map(float, lines[3].split()))) * scale
        c_vec = np.array(list(map(float, lines[4].split()))) * scale

        a = float(np.linalg.norm(a_vec))
        b = float(np.linalg.norm(b_vec))
        c = float(np.linalg.norm(c_vec))

        # 体积 = |det([a;b;c])|
        lattice = np.vstack([a_vec, b_vec, c_vec])  # 3x3
        V = float(abs(np.linalg.det(lattice)))

        if not np.isfinite([a, b, c, V]).all():
            return None

        return a, b, c, V
    except Exception:
        return None


# =========================
# 3) 读取 CSV 并构建数据
# =========================
df = pd.read_csv(CSV_PATH)

ta, tb, tc, tV = [], [], [], []
pa, pb, pc, pV = [], [], [], []

skipped = 0
for _, row in df.iterrows():
    tgt = abcV_from_poscar_str(row.get("target", None))
    pred = abcV_from_poscar_str(row.get("prediction", None))
    if tgt is None or pred is None:
        skipped += 1
        continue

    a_t, b_t, c_t, V_t = tgt
    a_p, b_p, c_p, V_p = pred

    ta.append(a_t); tb.append(b_t); tc.append(c_t); tV.append(V_t)
    pa.append(a_p); pb.append(b_p); pc.append(c_p); pV.append(V_p)

ta, tb, tc, tV = map(np.array, (ta, tb, tc, tV))
pa, pb, pc, pV = map(np.array, (pa, pb, pc, pV))

print(f"Total={len(df)}, Used={len(ta)}, Skipped={skipped}")

# R^2
r2_a = r2_score(ta, pa) if len(ta) > 0 else np.nan
r2_b = r2_score(tb, pb) if len(tb) > 0 else np.nan
r2_c = r2_score(tc, pc) if len(tc) > 0 else np.nan
r2_V = r2_score(tV, pV) if len(tV) > 0 else np.nan


# =========================
# 4) 绘图（2x2）
# =========================
def scatter_panel(ax, x, y, title, xlabel, ylabel, r2, lim=None):
    ax.scatter(x, y, s=10, alpha=0.7, edgecolors="none")

    # 对角线 y=x
    if lim is None:
        lo = min(np.min(x), np.min(y))
        hi = max(np.max(x), np.max(y))
        pad = 0.02 * (hi - lo + 1e-9)
        lo, hi = lo - pad, hi + pad
    else:
        lo, hi = lim

    ax.plot([lo, hi], [lo, hi], linewidth=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.text(
        0.5, 0.92, f"$R^2$: {r2:.2f}" if np.isfinite(r2) else "$R^2$: nan",
        transform=ax.transAxes, ha="center", va="center", fontsize=12
    )


# 你想完全“像 paper”那样固定范围的话可以用下面这些：
# a/b/c: 0~20, V: 0~1000
LIM_ABC = (0, 20)
LIM_V = (0, 1000)

fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=200)

# 背景颜色（可选）：paper 是浅绿色底
# fig.patch.set_facecolor("#dfead1")
# for ax in axes.ravel():
#     ax.set_facecolor("#dfead1")

scatter_panel(
    axes[0, 0], ta, pa,
    "(a) Lattice const. (x)",
    "Target a (Å)", "Pred. a (Å)",
    r2_a, lim=LIM_ABC
)

scatter_panel(
    axes[0, 1], tb, pb,
    "(b) Lattice const. (y)",
    "Target b (Å)", "Pred. b (Å)",
    r2_b, lim=LIM_ABC
)

scatter_panel(
    axes[1, 0], tc, pc,
    "(c) Lattice const. (z)",
    "Target c (Å)", "Pred. c (Å)",
    r2_c, lim=LIM_ABC
)

scatter_panel(
    axes[1, 1], tV, pV,
    "(d) Volume",
    "Target vol. (Å$^3$)", "Pred. vol. (Å$^3$)",
    r2_V, lim=LIM_V
)


fig.suptitle(
    "Prediction of Lattice Constants and Volume (Qwen_180)",
    fontsize=16
)

plt.tight_layout()
plt.savefig(FIG_PATH, bbox_inches="tight")
print("Saved figure to:", FIG_PATH)
# plt.show()
