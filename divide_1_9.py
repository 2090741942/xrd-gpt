import os
import shutil
import pandas as pd

SRC_ROOT = "/workspace/xrd-gpt/vasp_data_0.3_300"
DST_ROOT = "/workspace/xrd-gpt/vasp_data_0.3_300_1_9"

SUBDIRS = [
    ("id_prop", ""),
    ("id_prop_half", "_half"),
    ("id_prop_quarter", "_quarter"),
]

os.makedirs(DST_ROOT, exist_ok=True)

for subdir, suffix in SUBDIRS:
    src_dir = os.path.join(SRC_ROOT, subdir)
    dst_dir = os.path.join(DST_ROOT, subdir)
    os.makedirs(dst_dir, exist_ok=True)

    train_path = os.path.join(src_dir, f"id_prop_train{suffix}.csv")
    val_path   = os.path.join(src_dir, f"id_prop_val{suffix}.csv")
    test_path  = os.path.join(src_dir, f"id_prop_test{suffix}.csv")

    # 读 train + val
    train_df = pd.read_csv(train_path, header=None)
    val_df   = pd.read_csv(val_path, header=None)

    merged_train_df = pd.concat([train_df, val_df], ignore_index=True)

    # 写新的 train
    merged_train_df.to_csv(
        os.path.join(dst_dir, f"id_prop_train{suffix}.csv"),
        index=False,
        header=False,
    )

    # test 原样拷贝
    shutil.copy(
        test_path,
        os.path.join(dst_dir, f"id_prop_test{suffix}.csv"),
    )

    print(f"[OK] {subdir}: train+val -> train (1:9), test copied")


