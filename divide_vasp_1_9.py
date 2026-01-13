import os
import shutil
import pandas as pd

SRC = "/workspace/xrd-gpt/vasp_data_0.3_300"
DST = "/workspace/xrd-gpt/vasp_data_0.3_300_1_9"

# (suffix, train_dir, val_dir, test_dir)
SPECS = [
    ("",        "train",         "val",         "test"),
    ("_half",   "train_half",    "val_half",    "test_half"),
    ("_quarter","train_quarter", "val_quarter", "test_quarter"),
]

def ensure(p):
    os.makedirs(p, exist_ok=True)

def merge_csv(train_csv, val_csv, out_csv):
    df_t = pd.read_csv(train_csv, header=None)
    df_v = pd.read_csv(val_csv, header=None)
    pd.concat([df_t, df_v], ignore_index=True).to_csv(
        out_csv, index=False, header=False
    )

def copy_dir_contents(src, dst):
    ensure(dst)
    for name in os.listdir(src):
        s = os.path.join(src, name)
        d = os.path.join(dst, name)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

def replace_dir(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

def main():
    ensure(DST)

    for suffix, train_d, val_d, test_d in SPECS:
        # ===== CSV =====
        src_csv = os.path.join(SRC, f"id_prop{suffix}")
        dst_csv = os.path.join(DST, f"id_prop{suffix}")
        ensure(dst_csv)

        merge_csv(
            os.path.join(src_csv, f"id_prop_train{suffix}.csv"),
            os.path.join(src_csv, f"id_prop_val{suffix}.csv"),
            os.path.join(dst_csv, f"id_prop_train{suffix}.csv"),
        )

        shutil.copy2(
            os.path.join(src_csv, f"id_prop_test{suffix}.csv"),
            os.path.join(dst_csv, f"id_prop_test{suffix}.csv"),
        )

        # ===== VASP folders =====
        copy_dir_contents(
            os.path.join(SRC, train_d),
            os.path.join(DST, train_d),
        )
        copy_dir_contents(
            os.path.join(SRC, val_d),
            os.path.join(DST, train_d),
        )

        replace_dir(
            os.path.join(SRC, test_d),
            os.path.join(DST, test_d),
        )

        print(f"[OK] {train_d} + {val_d} -> {train_d}, {test_d} copied")

if __name__ == "__main__":
    main()
