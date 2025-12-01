import torch
import json
import os
from jarvis.db.jsonutils import loadjson
from typing import Optional
from typing import Literal
import pprint
from pydantic_settings import BaseSettings
import sys

# 获取当前脚本所在的文件夹（外层 atomgpt）
current_dir = os.path.dirname(os.path.abspath(__file__))

# 把内层 atomgpt 文件夹加入 Python 模块搜索路径
sys.path.append(os.path.join(current_dir, "atomgpt"))
from inverse_models.loader import FastLanguageModel

class TrainingPropConfig(BaseSettings):
    """Training config defaults and validation."""

    # id_prop_path: Optional[str] = "atomgpt/examples/inverse_model/id_prop.csv"

    id_prop_path_train: Optional[str] = "atomgpt/examples/inverse_model/id_prop.csv"
    id_prop_path_validation: Optional[str] = "atomgpt/examples/inverse_model/id_prop.csv"


    # "id_prop_path_train": "/data/atomgpt/atomgpt/DiffractGPT_stratified/id_prop_train.csv",
    # "id_prop_path_validation": "/data/atomgpt/atomgpt/DiffractGPT_stratified/id_prop_val.csv",

    prefix: str = "atomgpt_run"
    model_name: str = "knc6/atomgpt_mistral_tc_supercon"
    batch_size: int = 2
    num_epochs: int = 2
    # logging_steps: int = 1
    dataset_num_proc: int = 2
    seed_val: int = 3407
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    num_train: Optional[int] = None
    num_test: Optional[int] = None
    test_ratio: Optional[float] = 0.1
    val_ratio: Optional[float] = 0.1
    model_save_path: str = "atomgpt_lora_model"
    lora_rank: Optional[int] = 16
    lora_alpha: Optional[int] = 16
    loss_type: str = "default"
    optim: str = "adamw_8bit"
    id_tag: str = "id"
    lr_scheduler_type: str = "linear"
    separator: str = ","
    prop: str = "Tc_supercon"
    output_dir: str = "outputs"
    csv_out: str = "AI-AtomGen-prop-dft_3d-test-rmse.csv"
    chem_info: Literal["none", "formula", "element_list", "element_dict"] = (
        "formula"
    )
    file_format: Literal["poscar", "xyz", "pdb"] = "poscar"
    save_strategy: Literal["epoch", "steps", "no"] = "steps"
    save_steps: int = 20
    save_total_limit: int = 2

    callback_samples: int = 2
    max_seq_length: int = (
        2048  # Choose any! We auto support RoPE Scaling internally!
    )
    dtype: Optional[str] = None
    # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit: bool = True
    # True  # Use 4bit quantization to reduce memory usage. Can be False.
    instruction: str = "Below is a description of a superconductor material."
    alpaca_prompt: str = (
        "### Instruction:\n{}\n### Input:\n{}\n### Output:\n{}"
    )
    output_prompt: str = (
        " Generate atomic structure description with lattice lengths, angles, coordinates and atom types."
    )
    # num_val: Optional[int] = 2
    hp_cfg_path: Optional[str] = "hp_search_config.json"
    # per_device_train_batch_size: int = 2
    # gradient_accumulation_steps: int = 4
    warmup_steps: int = 3
    warmup_ratio: float = 0.0
    logging_steps: int = 10


def load_model_test(path="", config=None):
    if config is None:
        config_file = os.path.join(path, "config.json")
        config = loadjson(config_file)
        config = TrainingPropConfig(**config)
        pprint.pprint(config.dict())
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=config.dtype,
        load_in_4bit=config.load_in_4bit,
    )

    model.load_adapter(path)

    FastLanguageModel.for_inference(model)
    return model, tokenizer, config

def main():
    model, tokenizer, config = load_model_test(
        path="/data/atomgpt/lora_model_difractgpt_1_2"
    )

    with open("/data/atomgpt/outputs_xrd_1_2/alpaca_prop_test.json", encoding="utf-8") as f:
        data = json.load(f)

    # print(data[0])

    """
    {'id': 'JVASP-73731', 
    'instruction': 'Below is a description of a material.', 
    'input': 'The chemical formula is MgBe2Sn . The  XRD is 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.65,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.67,0.0,0.0,0.0,0.0,0.26,0.0,0.0,0.0,0.0,0.57,0.0,0.0,0.0,0.0,0.0,0.31,0.0,0.0,0.0,0.74,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.34,0.0,0.0,0.0,0.2,0.0,0.0,0.0,0.13,0.0,0.0,0.0,0.1,0.0,0.0,0.21,0.0,0.0,0.0,0.2,0.0,0.0,0.17,0.09,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.35,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.11,0.06,0.0,0.07,0.0,0.0,0.0,0.0,0.0,0.0,0.09,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.08,0.0,0.0,0.1,0.0,0.0,0.0,0.08,0.0,0.09,0.09,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.06,0.0,0.0,0.0,0.07,0.0,0.0,0.0,0.08,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.16,0.08,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.09,0.0,0.07,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.08,0.0,0.0,0.07,0.0,0.0,0.0,0.0,0.1,0.0,0.0,0.08,0.0,0.0,0.0,0.06,0.0,0.0,0.0,0.0,0.0,0.06,0.0,0.0,0.0,0.26,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.09,0.0,0.0,0.14,0.29,0.1,0.0,0.0,0.0,0.0,0.22,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.32,0.0,0.0,0.0,0.0,0.37,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0. Generate atomic structure description with lattice lengths, angles, coordinates and atom types.', 
    'output': '3.51 3.51 5.41\n90 90 90\nMg 0.500 0.500 0.000\nBe 0.000 0.000 0.773\nBe 0.000 0.000 0.227\nSn 0.500 0.500 0.500'}
    
    """

    data = data[:10]
    # eos_id = tokenizer.convert_tokens_to_ids("</s>")

    # print("eos_token:", tokenizer.eos_token)                  # 真实的 EOS 文本
    # print("eos_token_id:", tokenizer.eos_token_id)            # 真实的 EOS id
    # print("ids('</s>'):", tokenizer.encode("</s>", add_special_tokens=False))
    # print("ids('<|end_of_text|>'):", tokenizer.encode("<|end_of_text|>", add_special_tokens=False))
    # print("ids('<|eot_id|>'):", tokenizer.encode("<|eot_id|>", add_special_tokens=False))


    for item in data:

        # "alpaca_prompt": "### Instruction:\n{}\n### Input:\n{}\n### Output:\n{}"
        prompt = config.alpaca_prompt.format(item["instruction"], item["input"], "")

        input = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True, max_length=config.max_seq_length).to("cuda")

        with torch.no_grad():
            output = model.generate(
                **input,
                max_new_tokens=config.max_seq_length,
                use_cache=True,
                # eos_token_id=eos_id, 
            )
        
        text = tokenizer.decode(output[0], skip_special_tokens=True)

        with open("/data/atomgpt/example_to_see.csv", "a", encoding="utf-8") as file:
            file.write(f"=========={item['id']}==========\n")
            file.write(text.split("</s>")[0] + "\n")
            file.write("=" * 10 + "target" + "=" * 10 + "\n")
            file.write(item["output"] + "\n\n")

        print(f"=========={item['id']}==========")
        print(text.split("</s>")[0])
        print("=" * 10 + "target" + "=" * 10)
        print(item["output"])



if __name__ == "__main__":
    main()
    