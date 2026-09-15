# -*- coding: utf-8 -*-
"""QLoRA 微调脚本：Qwen2.5-1.5B-Instruct + peft。

工程要点（面试可讲）：
- **4bit 量化加载（NF4）**：1.5B 模型显存从 ~6G 降到 ~2G，4090 上跑得很轻松
- **LoRA 只训适配器**：可训练参数占全量的 ~1%，训练快、产物小（几十 MB）
- **只对回答计算 loss**（prompt 段 label 置 -100）：避免模型去学"复述问题"
- **固定种子**：数据与训练都可复现

用法（AutoDL 上）：
    python train_lora.py --model Qwen/Qwen2.5-1.5B-Instruct --epochs 3
产物：
    outputs/lora-adapter/   LoRA 适配器（可单独加载）
"""
from __future__ import annotations

import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          DataCollatorForSeq2Seq, Trainer, TrainingArguments)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="QLoRA 微调")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--train-file", default="data/train.jsonl")
    ap.add_argument("--output-dir", default="outputs/lora-adapter")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()



def _check_model_path(model: str) -> None:
    """本地路径但不存在时，给人话提示（否则 transformers 会把它当在线仓库名报一长串栈）。"""
    import os
    looks_local = model.startswith("/") or model.startswith("./") or model.startswith("~")
    if looks_local and not os.path.isdir(os.path.expanduser(model)):
        raise SystemExit(
            f"\n[!] 模型路径不存在：{model}\n"
            f"    请先下载基座模型（国内推荐 ModelScope）：\n"
            f"      pip install modelscope\n"
            f"      python -c \"from modelscope import snapshot_download; "
            f"print(snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', cache_dir='/root/autodl-tmp/models'))\"\n"
            f"    然后用打印出来的路径跑本脚本。\n"
        )


def main() -> None:
    a = parse_args()
    _check_model_path(a.model)
    torch.manual_seed(a.seed)

    # ---------- 1. 分词器 & 数据 ----------
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    raw = load_dataset("json", data_files=a.train_file, split="train")

    def tokenize(ex):
        """prompt 段 label 置 -100：只让模型学「怎么答」，不学「怎么复述问题」。"""
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": ex["instruction"]}],
            tokenize=False, add_generation_prompt=True,
        )
        answer = ex["output"] + tok.eos_token
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        a_ids = tok(answer, add_special_tokens=False)["input_ids"]
        ids = (p_ids + a_ids)[: a.max_len]
        labels = ([-100] * len(p_ids) + a_ids)[: a.max_len]
        return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}

    ds = raw.map(tokenize, remove_columns=raw.column_names)
    print(f"训练样本：{len(ds)} 条，示例长度 {len(ds[0]['input_ids'])} tokens")

    # ---------- 2. 4bit 量化加载 ----------
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        a.model, quantization_config=bnb, device_map="auto", trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # ---------- 3. LoRA ----------
    lora = LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_r * 2, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()     # 打印可训练参数占比（一般是 ~1%）

    # ---------- 4. 训练 ----------
    args = TrainingArguments(
        output_dir=a.output_dir + "/ckpt",
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        bf16=torch.cuda.is_bf16_supported(),
        report_to=[],                       # 不接外部追踪
        seed=a.seed,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=ds,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
    )
    trainer.train()

    os.makedirs(a.output_dir, exist_ok=True)
    model.save_pretrained(a.output_dir)
    tok.save_pretrained(a.output_dir)
    print(f"\nLoRA 适配器已保存到：{a.output_dir}")


if __name__ == "__main__":
    main()
