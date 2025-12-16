import pandas as pd
import numpy as np
from io import StringIO

def parse_lattice_constants_from_poscar(poscar_str: str):
    if not isinstance(poscar_str, str):
        return None

    # 关键一行：把字面量 "\n" 还原成真实换行
    poscar_str = poscar_str.replace("\\n", "\n")

    lines = [l.strip() for l in poscar_str.splitlines() if l.strip()]
    if len(lines) < 5:
        return None

    # try:
    #     scale = float(lines[1])
    #     a_vec = np.array(list(map(float, lines[2].split()))) * scale
    #     b_vec = np.array(list(map(float, lines[3].split()))) * scale
    #     c_vec = np.array(list(map(float, lines[4].split()))) * scale

    #     return (
    #         float(np.linalg.norm(a_vec)),
    #         float(np.linalg.norm(b_vec)),
    #         float(np.linalg.norm(c_vec)),
    #     )
    # except Exception:
    #     return None

    try:
        scale = float(lines[1])
        a_vec = np.array(list(map(float, lines[2].split()))) * scale
        b_vec = np.array(list(map(float, lines[3].split()))) * scale
        c_vec = np.array(list(map(float, lines[4].split()))) * scale

        a = float(np.linalg.norm(a_vec))
        b = float(np.linalg.norm(b_vec))
        c = float(np.linalg.norm(c_vec))

        # 关键：过滤 nan/inf
        if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(c)):
            return None

        return a, b, c
    except Exception:
        return None





def compute_abc_mae(csv_path: str):
    df = pd.read_csv(csv_path)

    a_err, b_err, c_err = [], [], []
    skipped = 0

    for _, row in df.iterrows():
        tgt = parse_lattice_constants_from_poscar(row["target"])
        pred = parse_lattice_constants_from_poscar(row["prediction"])

        if tgt is None or pred is None:
            skipped += 1
            continue

        a_t, b_t, c_t = tgt
        a_p, b_p, c_p = pred

        a_err.append(abs(a_p - a_t))
        # b_err.append(abs(b_p - b_t))
        err_b = abs(b_p - b_t)
        # if not np.isfinite(err_b):
        #     print("BAD b:", row["id"])
        #     print("target b:", b_t, "pred b:", b_p)
        #     print("target lattice lines:", row["target"].replace("\\n","\n").splitlines()[0:6])
        #     print("pred lattice lines:", row["prediction"].replace("\\n","\n").splitlines()[0:6])
        #     break
        b_err.append(err_b)

        c_err.append(abs(c_p - c_t))

    return {
        "MAE_a": np.mean(a_err),
        "MAE_b": np.mean(b_err),
        "MAE_c": np.mean(c_err),
        "N_valid": len(a_err),
        "N_skipped": skipped,
    }


if __name__ == "__main__":
    result = compute_abc_mae("/workspace/generation_result/AI-AtomGen-prop-dft_3d-test-rmse_quarter90_qwen2.5.csv")
    for k, v in result.items():
        print(f"{v}")
        # print(f"{k}: {v}")
