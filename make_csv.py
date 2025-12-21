import json
import csv

json_path = "/workspace/atomgpt/outputs_xrd_quarter90_0.3/alpaca_prop_test.json"
csv_path = "/workspace/pred_list_inverse_90_0.3.csv"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)   # data 是一个 list

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    for item in data:
        writer.writerow([item.get("input", "")])
    # for item in data:
    #     s = item.get("input", "")
    #     if s.startswith('"') and s.endswith('"'):
    #         s = s[1:-1]
    #     writer.writerow([s])


"""
在inverse_predict里使用如下代码读取
with open("/workspace/output.csv", "r", encoding="utf-8", newline="") as f:
    reader = csv.reader(f)
    lines = [line[0] for line in reader]
"""