import os
import csv
import pandas as pd

from jarvis.db.figshare import data
from jarvis.core.atoms import Atoms
from pymatgen.io.vasp import Poscar


# ================ 配置区 ================
SPLITS = {
    "train": "id_prop_train.csv",
    "val":   "id_prop_val.csv",
    "test":  "id_prop_test.csv",
    # 如果你有预测集，就加一行：
    # "predict": "id_prop_predict.csv",
}

OUT_ROOT = "vasp_data"   # 输出根目录
JID_COL = 0              # jid 在 csv 的第 0 列
# =======================================


def load_jids_from_csv(csv_path: str, jid_col: int = 0) -> list[str]:
    """从 CSV 第一列读取 jid（无表头时最常见）"""
    jids = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            jid = row[jid_col].strip()
            if jid:
                jids.append(jid)
    return jids


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)

    print("Loading JARVIS dft_3d dataset ...")
    df = pd.DataFrame(data("dft_3d")).set_index("jid")

    for split, csv_path in SPLITS.items():
        split_dir = os.path.join(OUT_ROOT, split)
        os.makedirs(split_dir, exist_ok=True)

        jids = load_jids_from_csv(csv_path, JID_COL)
        print(f"\n[{split}] jids = {len(jids)} from {csv_path}")

        missing = 0
        written = 0

        for jid in jids:
            if jid not in df.index:
                print(f"[WARN] jid not found in dft_3d: {jid}")
                missing += 1
                continue

            atoms = Atoms.from_dict(df.loc[jid, "atoms"])
            pmg_struct = atoms.pymatgen_converter()

            # 文件名就是 jid，例如 vasp_data/train/JVASP-12345
            out_path = os.path.join(split_dir, jid)

            # 写 POSCAR 内容到这个文件（不是 POSCAR 这个名字）
            Poscar(pmg_struct).write_file(out_path)

            written += 1

        print(f"[{split}] written={written}, missing={missing}")

    print("\nDone.")


if __name__ == "__main__":
    main()
