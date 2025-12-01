import pandas as pd

data_full = pd.read_csv("/workspace/xrd-gpt/jarvis_data/id_prop/id_prop_train.csv", encoding="utf-8")
data_quarter = pd.read_csv("/workspace/xrd-gpt/jarvis_data/id_prop_quarter/id_prop_train_quarter.csv", encoding="utf-8")

print(data_full.info)
print(data_quarter.info)
