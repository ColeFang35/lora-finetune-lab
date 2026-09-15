# -*- coding: utf-8 -*-
"""合成指令数据集生成器（差旅 / 客服域）。

为什么要有这个脚本：
- 后训练的第一步不是调参，是**数据**。这里用一个可复现的模板 + 组合式生成，
  产出「用户提问 → 期望回答」的指令对，覆盖查订单 / 查物流 / 退款 / 改签 / 报销 / 会员等意图。
- 真实生产里的做法是：**线上 bad case 挖掘 + 强模型合成 + 规则/裁判过滤 + 去重**；
  这个脚本把"合成 + 结构约束"这一段做出来，替换掉模板源即可接真实数据。

产物：
  data/train.jsonl  训练集（约 需要多少条就生成多少条）
  data/test.jsonl   留出测试集（用于微调前后对比）
格式：{"instruction": ..., "input": ..., "output": ...}
"""
from __future__ import annotations

import argparse
import json
import os
import random

random.seed(20260915)   # 固定种子 → 数据可复现

CITIES = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "西安", "南京", "青岛"]
ORDERS = ["SO20260810001", "SO20260812003", "SO20260815005", "SO20260901007", "SO20260903011"]
STATUS = ["已发货", "待发货", "待付款", "已签收"]
CARRIERS = ["云雀速运", "顺丰速运", "京东物流"]

TEMPLATES: list[tuple[str, list[str], str]] = [
    # (意图, 问题模板, 回答模板)
    ("查订单",
     ["帮我查一下订单 {o}", "订单 {o} 现在什么状态", "查下 {o} 这个单子"],
     "订单 {o} 当前状态为「{s}」，如需了解物流或办理售后，请告诉我。"),
    ("查物流",
     ["查一下 {o} 的物流到哪了", "{o} 什么时候能到", "帮我看看 {o} 的快递"],
     "订单 {o} 由{c}承运，状态为「{s}」。如需查询具体轨迹，我可以继续帮您拉取最新节点。"),
    ("退款咨询",
     ["{o} 能退款吗", "我想退 {o} 这个订单", "{o} 支持退货吗"],
     "订单 {o} 的状态是「{s}」。退款需先核查订单状态与时效，符合条件的订单可提交退款申请，"
     "原路退回一般 3-5 个工作日到账。"),
    ("改签/取消",
     ["{o} 能取消吗", "帮我取消 {o}", "{o} 我想改一下行程"],
     "订单 {o} 当前状态为「{s}」。未发货/未出票的订单可以取消；已出票的需按规则办理改签，"
     "可能产生手续费，需您确认后我再提交。"),
    ("差旅政策",
     ["{city}出差住宿报销标准是多少", "去{city}出差的差旅费怎么报", "{city}的差标是什么"],
     "前往{city}的出差标准：住宿按公司差旅制度执行，需在标准内选择酒店并保留发票；"
     "交通优先选择经济舱/二等座。具体额度请以最新差旅制度为准。"),
    ("申请审批",
     ["帮我提一个去{city}的出差申请", "我要申请去{city}出差", "帮我发起 {city} 的出差审批"],
     "已为您发起前往{city}的出差申请，审批流程为「直属上级 → 部门负责人」，"
     "时效承诺 24 小时内。审批进度可随时找我查询。"),
    ("会员咨询",
     ["我的会员等级是什么", "看看我的积分和优惠券", "会员有什么权益"],
     "您的账户信息需通过手机号后四位核验后查询。核验后我可以为您查看会员等级、积分余额与可用优惠券。"),
    ("转人工",
     ["我要转人工", "帮我找人工客服", "这里解决不了，转人工"],
     "已为您转接人工客服，工单已创建，专员会在工作时间内尽快联系您。"),
]


def _gen(n: int) -> list[dict]:
    rows: list[dict] = []
    intents = list(TEMPLATES)
    for _ in range(n):
        intent, qs, ans = random.choice(intents)
        o, city, s, c = (random.choice(ORDERS), random.choice(CITIES),
                         random.choice(STATUS), random.choice(CARRIERS))
        q = random.choice(qs).format(o=o, city=city)
        a = ans.format(o=o, city=city, s=s, c=c)
        rows.append({
            "instruction": q,
            "input": "",
            "output": a,
            "meta": {"intent": intent},          # 便于分类统计，训练时可忽略
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="生成合成指令数据集")
    ap.add_argument("--train", type=int, default=400, help="训练集条数")
    ap.add_argument("--test", type=int, default=60, help="测试集条数")
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    train = _gen(args.train)
    test = _gen(args.test)
    for name, rows in (("train.jsonl", train), ("test.jsonl", test)):
        path = os.path.join(args.out, name)
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"写入 {name}：{len(rows)} 条")
    # 意图分布（便于检查覆盖是否均衡）
    from collections import Counter
    print("意图分布：", dict(Counter(r["meta"]["intent"] for r in train).most_common()))


if __name__ == "__main__":
    main()
