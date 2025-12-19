import shutil
from pathlib import Path

import pandas as pd

# 项目根目录（本文件所在目录）
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "JARVIS_90"

# 三个 split 的配置：csv 文件名 + 对应 VASP 目录名
SPLITS = {
    "train": {
        "csv": "id_prop_train.csv",
        "vasp_dir": "training",
    },
    "val": {
        "csv": "id_prop_val.csv",
        "vasp_dir": "validation",
    },
    "test": {
        "csv": "id_prop_test.csv",
        "vasp_dir": "testing",
    },
}


def copy_vasp_files(df_subset: pd.DataFrame, vasp_src_dir: Path, subset_dir: Path):
    """把 df_subset 中第一列列出的 JVASP-* 文件复制到 subset_dir"""
    subset_dir.mkdir(parents=True, exist_ok=True)

    id_col = df_subset.columns[0]
    ids = df_subset[id_col].unique()

    for jvasp_id in ids:
        src = vasp_src_dir / str(jvasp_id)
        dst = subset_dir / str(jvasp_id)
        if src.is_file():
            shutil.copy2(src, dst)
        else:
            # 安全起见，万一 csv 里有但文件缺失，就打印一下
            print(f"  [WARN] VASP file not found: {src}")


def make_subsets_for_split(
    split_name: str,
    csv_name: str,
    vasp_dir_name: str,
    random_state: int = 42,
):
    csv_path = DATA_DIR / csv_name
    vasp_dir = DATA_DIR / vasp_dir_name

    print(f"\n=== Processing split: {split_name} ===")
    print(f"CSV : {csv_path}")
    print(f"VASP: {vasp_dir}")

    # 1) 读入 CSV（假设没有表头：第一列是 JVASP-id，后面是性质）
    df = pd.read_csv(csv_path, header=None)
    id_col = df.columns[0]
    print(f"  Original rows: {len(df)}")

    # 2) 获取实际存在的 JVASP 文件名（无扩展名）
    existing_ids = {p.name for p in vasp_dir.iterdir() if p.is_file()}
    print(f"  VASP files found: {len(existing_ids)}")

    # 3) 只保留在 VASP 目录里确实存在的 id
    df_filtered = df[df[id_col].isin(existing_ids)].reset_index(drop=True)
    n = len(df_filtered)
    print(f"  Rows after matching VASP files: {n}")

    if n == 0:
        print("  [SKIP] No matching rows for this split.")
        return

    # 4) 计算 1/2 和 1/4 的大小
    n_half = max(1, n // 2)
    n_quarter = max(1, n // 4)
    print(f"  Half size   : {n_half}")
    print(f"  Quarter size: {n_quarter}")

    # 5) 随机采样 1/2 和 1/4 的子集（独立采样）
    df_half = df_filtered.sample(n=n_half, random_state=random_state)
    df_quarter = df_filtered.sample(n=n_quarter, random_state=random_state + 1)

    # # 6) 为两个子集建立新目录
    # half_dir = DATA_DIR / f"{vasp_dir_name}_half"
    # quarter_dir = DATA_DIR / f"{vasp_dir_name}_quarter"

    # half_dir.mkdir(parents=True, exist_ok=True)
    # quarter_dir.mkdir(parents=True, exist_ok=True)

    # # 7) 在各自目录下写入对应 CSV
    # half_csv_path = half_dir / f"id_prop_{split_name}_half.csv"
    # quarter_csv_path = quarter_dir / f"id_prop_{split_name}_quarter.csv"

    # df_half.to_csv(half_csv_path, index=False, header=False)
    # df_quarter.to_csv(quarter_csv_path, index=False, header=False)

    # print(f"  Saved half CSV    -> {half_csv_path}")
    # print(f"  Saved quarter CSV -> {quarter_csv_path}")

    # 6) 为两个子集建立 VASP 目录（保持你现在的结构）
    half_dir = DATA_DIR / f"{vasp_dir_name}_half"
    quarter_dir = DATA_DIR / f"{vasp_dir_name}_quarter"

    half_dir.mkdir(parents=True, exist_ok=True)
    quarter_dir.mkdir(parents=True, exist_ok=True)

    # 7) CSV：只用两个总目录 id_prop_half / id_prop_quarter
    id_prop_half_dir = DATA_DIR / "id_prop_half"
    id_prop_quarter_dir = DATA_DIR / "id_prop_quarter"

    id_prop_half_dir.mkdir(parents=True, exist_ok=True)
    id_prop_quarter_dir.mkdir(parents=True, exist_ok=True)

    half_csv_path = id_prop_half_dir / f"id_prop_{split_name}_half.csv"
    quarter_csv_path = id_prop_quarter_dir / f"id_prop_{split_name}_quarter.csv"

    df_half.to_csv(half_csv_path, index=False, header=False)
    df_quarter.to_csv(quarter_csv_path, index=False, header=False)

    print(f"  Saved half CSV    -> {half_csv_path}")
    print(f"  Saved quarter CSV -> {quarter_csv_path}")

    # 8) 复制对应的 VASP 文件到新目录
    print(f"  Copying VASP files for HALF subset to: {half_dir}")
    copy_vasp_files(df_half, vasp_dir, half_dir)

    print(f"  Copying VASP files for QUARTER subset to: {quarter_dir}")
    copy_vasp_files(df_quarter, vasp_dir, quarter_dir)


def main():
    for split_name, cfg in SPLITS.items():
        make_subsets_for_split(
            split_name=split_name,
            csv_name=cfg["csv"],
            vasp_dir_name=cfg["vasp_dir"],
        )


if __name__ == "__main__":
    main()
