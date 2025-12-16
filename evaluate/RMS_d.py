import math
import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher


# =========================
# 1. 文件路径
# =========================
INPUT_CSV = "/workspace/generation_result/AI-AtomGen-prop-dft_3d-test-rmse_quarter90_qwen2.5.csv"
OUTPUT_CSV = "/workspace/rmsd_results_180_qwen.csv"


# =========================
# 2. POSCAR 字符串 → Structure
# =========================
def structure_from_poscar_string(poscar_str: str) -> Structure:
    if not isinstance(poscar_str, str):
        raise ValueError("POSCAR is not a string")

    # 关键：把 CSV 中的字面量 \\n 还原成真实换行
    poscar_str = poscar_str.replace("\\n", "\n").strip()

    if len(poscar_str) < 10:
        raise ValueError("POSCAR string too short")

    return Structure.from_str(poscar_str, fmt="poscar")


# =========================
# 3. RMS-d 计算（论文定义）
# =========================
def compute_rmsd_normalized(struct_pred, struct_gt, matcher):
    """
    RMS-d = RMS / (V/N)^(1/3)
    """
    if not matcher.fit(struct_pred, struct_gt):
        return None

    rms = matcher.get_rms_dist(struct_pred, struct_gt)
    if rms is None or math.isnan(rms) or math.isinf(rms):
        return None

    V = struct_gt.volume
    N = len(struct_gt)
    if V <= 0 or N <= 0:
        return None

    norm = (V / N) ** (1.0 / 3.0)
    if norm <= 0:
        return None

    return float(rms / norm)


# =========================
# 4. 主流程
# =========================
def main():
    df = pd.read_csv(INPUT_CSV)

    matcher = StructureMatcher(
        stol=0.5,
        angle_tol=10,
        ltol=0.3,
        primitive_cell=False,
        scale=True,
        attempt_supercell=False,
    )

    rows = []
    valid_rmsd = []

    for _, row in df.iterrows():
        jid = row["id"]
        rmsd_value = np.nan

        try:
            struct_gt = structure_from_poscar_string(row["target"])
            struct_pred = structure_from_poscar_string(row["prediction"])

            rmsd = compute_rmsd_normalized(struct_pred, struct_gt, matcher)
            if rmsd is not None:
                rmsd_value = rmsd
                valid_rmsd.append(rmsd)

        except Exception:
            # 解析失败 / 不 match / 非法结构 → NaN
            pass

        rows.append({
            "id": jid,
            "RMS-d": rmsd_value
        })

    # 平均值（只对有效样本）
    avg_rmsd = float(np.mean(valid_rmsd)) if len(valid_rmsd) > 0 else np.nan
    rows.append({
        "id": "AVERAGE",
        "RMS-d": avg_rmsd
    })

    out_df = pd.DataFrame(rows, columns=["id", "RMS-d"])
    out_df.to_csv(OUTPUT_CSV, index=False)

    print("Done.")
    print(f"Total samples : {len(df)}")
    print(f"Matched       : {len(valid_rmsd)}")
    print(f"Skipped       : {len(df) - len(valid_rmsd)}")
    print(f"Average RMS-d : {avg_rmsd}")


if __name__ == "__main__":
    main()
