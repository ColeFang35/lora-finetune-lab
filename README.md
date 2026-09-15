# LoRA 微调实验台（Qwen2.5-1.5B + QLoRA）

一次**真实可跑**的后训练实验：在自建的指令数据集上做 **QLoRA 微调**，并用**同一测试集、同一解码参数**对比微调前后的效果。

- **数据**：`data/build_dataset.py` 可复现地生成差旅/客服域指令数据（合成数据构建）
- **训练**：4bit NF4 量化 + LoRA，只训 ~1% 参数；**只对回答算 loss**（prompt 段 label 置 -100）
- **评测**：格式合规率 / 实体命中率 / 意图合理率三个可离线计算的指标，直接回答"微调到底有没有用"

## 一、AutoDL 上跑一遍（约 1–2 小时）

### 1. 开机器
- **GPU**：RTX 4090（24G）—— 1.5B 模型 QLoRA 只吃 ~8G，4090 绰绰有余
- **镜像**：`PyTorch 2.x` + `CUDA 12.x`（例如 `PyTorch 2.3.0 / CUDA 12.1`）
- **数据盘**：20G 够用（基座模型 ~3G + 数据 + 输出）

### 2. 把代码传上去
三种任选：
```bash
# ① 直接 git clone（如果你放到了 GitHub）
git clone <你的仓库地址> && cd lora-finetune-lab

# ② 本地上传（AutoDL 控制台有「上传」或 JupyterLab 拖拽）
# ③ scp
scp -P <端口> -r ./lora-finetune-lab root@<AutoDL地址>:/root/
```

### 3. 装依赖
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 4. 下载基座模型（国内建议用 ModelScope，比 HF 快很多）
```bash
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', cache_dir='/root/autodl-tmp/models')"
```
然后用本地路径跑（把 `--model` 换成 `/root/autodl-tmp/models/Qwen/Qwen2.5-1.5B-Instruct`）。

### 5. 生成数据 → 微调 → 评测
```bash
python data/build_dataset.py                       # 生成 train/test（可复现）

python train_lora.py \
  --model /root/autodl-tmp/models/Qwen/Qwen2.5-1.5B-Instruct \
  --epochs 3

python evaluate.py \
  --base /root/autodl-tmp/models/Qwen/Qwen2.5-1.5B-Instruct \
  --adapter outputs/lora-adapter
```

### 6. 拿走产物
- `outputs/lora-adapter/`（几十 MB，可单独加载）
- `reports/eval_compare.json`（**微调前后指标对比 + 样例**）—— 把这个贴进简历/面试

## 二、脚本在做什么

| 文件 | 要点 |
|---|---|
| `data/build_dataset.py` | 模板 + 组合式生成指令数据；固定随机种子 → 可复现；真实生产里把"模板"换成「线上 bad case 挖掘 + 强模型合成 + 裁判过滤」 |
| `train_lora.py` | 4bit NF4 量化加载；LoRA（r=8，alpha=16，7 个投影层）；**prompt 段 labels=-100**；cosine + warmup；bf16 |
| `evaluate.py` | 贪心解码保证可复现；**实体命中率**（订单号/城市/状态/承运商）比 BLEU 更贴业务；退化检测 |

## 三、预期结果与常见坑

**预期**：微调后 **实体命中率** 与 **意图合理率** 明显上升（基座不知道你的业务实体与话术）；
格式合规率通常本来就不低。

**常见坑**：
| 现象 | 处理 |
|---|---|
| `CUDA out of memory` | 调小 `--batch-size`，或用 `--grad-accum` 保持等效 batch |
| `bitsandbytes` 报错 | 确认 CUDA 版本与 bnb 版本匹配；`pip install -U bitsandbytes` |
| 生成重复退化 | `evaluate.py` 已加 `repetition_penalty`；训练端可加大数据量或降 lr |
| 微调后"变傻" | 数据量小/epoch 多 → 过拟合。看评测里是否只有训练集分布上变好 |

## 四、可继续做的（面试能延伸的方向）
- 换更大基座（Qwen2.5-7B）看 LoRA 收益是否变化
- 加 **DPO / 偏好优化**：用同一评测集构造 chosen/rejected 对
- 接 **LLM-as-a-Judge** 做开放式质量打分（配套项目 `llm-judge-eval`）
