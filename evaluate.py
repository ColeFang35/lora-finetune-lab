# -*- coding: utf-8 -*-
"""微调前后对比评测：同一测试集，同一解码参数，只换模型。

指标设计（都能离线算，不依赖外部裁判）：
- **格式合规率**：回答非空、长度在合理区间、无重复退化
- **实体命中率**：生成回答里是否包含参考答案中的关键实体（订单号 / 城市 / 状态 / 承运商）
  —— 直接衡量"有没有答到点子上"，比 BLEU/ROUGE 更贴业务
- **意图合理率**：回答是否落在该意图应有的语义范围（用关键词集合近似）
（可选）--judge-api：接一个 OpenAI 兼容接口做 LLM 裁判，给出 1~5 分

用法（AutoDL）：
    python evaluate.py --base Qwen/Qwen2.5-1.5B-Instruct --adapter outputs/lora-adapter
"""
from __future__ import annotations

import argparse
import json
import os
import re
from statistics import mean

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

CITIES = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "西安", "南京", "青岛"]
STATUS = ["已发货", "待发货", "待付款", "已签收"]
CARRIERS = ["云雀速运", "顺丰速运", "京东物流"]
ORDER_RE = re.compile(r"SO\d{8,}")
INTENT_KEYS = {
    "查订单": ["订单", "状态"], "查物流": ["物流", "承运", "状态"],
    "退款咨询": ["退款", "退回", "退货"], "改签/取消": ["取消", "改签", "手续费"],
    "差旅政策": ["住宿", "差标", "发票", "交通"], "申请审批": ["申请", "审批"],
    "会员咨询": ["会员", "积分", "优惠券", "手机号"], "转人工": ["人工", "工单"],
}


def parse_args():
    ap = argparse.ArgumentParser(description="微调前后对比评测")
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None, help="LoRA 适配器目录；不传则只评基座")
    ap.add_argument("--test-file", default="data/test.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--limit", type=int, default=60, help="评测样本数（控制耗时）")
    ap.add_argument("--out", default="reports/eval_compare.json")
    return ap.parse_args()


def load_model(base: str, adapter: str | None):
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # transformers 5.x 把 torch_dtype 改名为 dtype → 做版本兼容
    kw = dict(device_map="auto", trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, **kw)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


@torch.no_grad()
def generate(model, tok, instruction: str, max_new_tokens: int) -> str:
    msgs = [{"role": "user", "content": instruction}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,   # 贪心解码，保证可复现
                         repetition_penalty=1.1, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def score_one(pred: str, ref: str, intent: str) -> dict:
    """单条打分：格式 / 实体 / 意图关键词。"""
    ok_format = bool(pred) and 8 <= len(pred) <= 600 and not _degenerate(pred)
    ents = ORDER_RE.findall(ref) + [c for c in CITIES if c in ref] \
        + [s for s in STATUS if s in ref] + [c for c in CARRIERS if c in ref]
    hit = [e for e in set(ents) if e in pred]
    ent_rate = len(hit) / len(set(ents)) if ents else 1.0
    keys = INTENT_KEYS.get(intent, [])
    key_rate = (sum(1 for k in keys if k in pred) / len(keys)) if keys else 1.0
    return {"format": float(ok_format), "entity": round(ent_rate, 3), "intent": round(key_rate, 3)}


def _degenerate(t: str) -> bool:
    """检测退化：整段重复同一短语。"""
    if len(t) < 20:
        return False
    for n in (4, 6, 8):
        for i in range(0, min(len(t) - 2 * n, 200)):
            if t[i:i + n] == t[i + n:i + 2 * n] == t[i + 2 * n:i + 3 * n]:
                return True
    return False


def evaluate(model, tok, cases: list[dict], max_new_tokens: int) -> tuple[dict, list[dict]]:
    rows = []
    for c in cases:
        pred = generate(model, tok, c["instruction"], max_new_tokens)
        s = score_one(pred, c["output"], c.get("meta", {}).get("intent", ""))
        rows.append({"instruction": c["instruction"], "ref": c["output"], "pred": pred, **s})
    agg = {k: round(mean(r[k] for r in rows), 3) for k in ("format", "entity", "intent")}
    agg["overall"] = round(mean(agg.values()), 3)
    return agg, rows


def main() -> None:
    a = parse_args()
    cases = [json.loads(l) for l in open(a.test_file, encoding="utf-8")][: a.limit]
    print(f"测试集：{len(cases)} 条\n")

    results: dict[str, dict] = {}
    samples: dict[str, list] = {}

    for tag, adapter in (("base", None), ("lora", a.adapter)):
        if tag == "lora" and not adapter:
            continue
        model, tok = load_model(a.base, adapter)
        agg, rows = evaluate(model, tok, cases, a.max_new_tokens)
        results[tag] = agg
        samples[tag] = rows[:3]
        print(f"[{tag:>4}] 格式合规 {agg['format']:.3f} | 实体命中 {agg['entity']:.3f} | "
              f"意图合理 {agg['intent']:.3f} | 综合 {agg['overall']:.3f}")
        del model
        torch.cuda.empty_cache()

    if "base" in results and "lora" in results:
        d = {k: round(results["lora"][k] - results["base"][k], 3) for k in results["base"]}
        print(f"\n微调前后提升：{d}")
        results["delta"] = d

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"metrics": results, "samples": samples}, f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入：{a.out}")
    print("\n=== 样例对比（微调前 → 微调后）===")
    if "base" in samples and "lora" in samples:
        for b, l in zip(samples["base"], samples["lora"]):
            print(f"\n问：{b['instruction']}")
            print(f"  微调前：{b['pred'][:80]}")
            print(f"  微调后：{l['pred'][:80]}")


if __name__ == "__main__":
    main()
