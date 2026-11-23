# GPU 环境准备指南

> 场景：本机为 `torch 2.9.1+cpu`、无 CUDA，跑不动 Qwen2.5-0.5B 的 1,200 段生成。
> 本文给出可落地的获取 GPU 的途径、成本与完整命令。

---

## 1. 先看需求：其实很小

| 项目 | 需求 | 说明 |
|---|---|---|
| 显存 | **≥ 2 GB**（推荐 8 GB 以上） | 0.5B fp16 权重约 1 GB，加 KV cache 余量 |
| 磁盘 | ~5 GB | 模型 1 GB + 依赖 + 输出 |
| 时长 | 单流约 1—2 小时；批处理 20—40 分钟（估算） | 1,200 段 × 200 token ≈ 24 万 token |
| 预算 | 免费 ~ 10 元 | 0.5B 是极小模型，不需要 A100 |

结论：**T4 / 3090 / 4060 这一档就够**，不需要抢 A100。

---

## 2. 三条路径（按推荐顺序）

### 路线 A：学校 / 实验室（首选，免费）

1. 问导师或实验室师兄：有没有 GPU 服务器、是否给本科生开账号；
2. 学院机房 / 计算中心常有共享 GPU（Slurm 或 Docker 提交任务）；
3. 优势：免费、无需上传数据、可长期跑。

**如果走这条，只需问三件事**：有没有 GPU、怎么申请账号、能不能装自己的 Python 包。

### 路线 B：国内免费云（次选，零成本）

| 平台 | 免费额度 | 网络 | 备注 |
|---|---|---|---|
| 百度 AI Studio | 每日签到送算力点，可开 V100/A100 | 国内直连 | 需实名认证，Notebook 形式，最省事 |
| 阿里 魔搭 ModelScope | 免费 Notebook GPU 时长 | 国内直连 | 与 HF 模型生态兼容，模型下载快 |
| 阿里云 PAI-DSW | 新用户免费试用 | 国内 | 有额度限制 |
| Kaggle / Colab | T4 免费（每周 30h / 每日若干小时） | 需国际网络 | 国内不稳定，仅作备选 |

**推荐 AI Studio 或魔搭**：不用付钱、不用翻墙、有 GPU 就能跑。

### 路线 C：按小时租（最省心，几块钱）

| 平台 | 常见卡型 | 价格（参考） |
|---|---|---|
| AutoDL | RTX 3090 / 4090 | 约 1—2 元/小时 |
| 恒源云、蓝耘、矩池云 | 3090 / 4090 / T4 | 约 1—3 元/小时 |

按 2 小时算，**总花费约 3—5 元**。适合“不想折腾、今天就要结果”。

> 学生建议：先试 B（免费），B 排队/限流时再用 C（几块钱买时间）。

---

## 3. 上去之后怎么装

以下命令 Ubuntu + Python 3.10/3.11 通用。

```bash
# 1) 克隆仓库
git clone https://github.com/lcyspring/WatermarkSys_LLMText.git
cd WatermarkSys_LLMText/WatermarkSys_LLMText

# 2) 装依赖（GPU 版 torch 按 CUDA 版本选，cu121 兼容性最好）
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers fastapi uvicorn jinja2 pyyaml numpy pytest httpx

# 3) 确认能看到 GPU
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

国内下载慢时加镜像：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 4. 模型从哪来（关键，别用 GitHub 直连 HF）

**方式一：HF 镜像（推荐）**

```bash
export HF_ENDPOINT=https://hf-mirror.com
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct'); print('ok')"
```

**方式二：ModelScope（国内最快，阿里云/魔搭实例首选）**

```bash
pip install modelscope
python -c "from modelscope import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-0.5B-Instruct'))"
```

> ModelScope 下载后是本地目录，把 `--model-id` 改成该目录路径即可（如 `/root/.cache/modelscope/hub/Qwen/Qwen2.5-0.5B-Instruct`）。

## 5. 跑实验（注意 `--vocab`）

Qwen2.5 系列**词表是 151,936**，不是默认的 2,048；传错会导致绿名单划分错误、z 被稀释。

```bash
# 先 20 条小规模，确认耗时与正确性
python scripts/04_generate.py --mode hf --model-id Qwen/Qwen2.5-0.5B-Instruct \
       --deltas 0,0.5,1,2 --limit 20 --max-new-tokens 128 \
       --vocab 151936 --out outputs/hf_qwen_20.jsonl --fresh

python scripts/06_detect.py --input outputs/hf_qwen_20.jsonl --vocab 151936

# 确认无误后再全量（60 提示 × 4 档 = 240 条；如要 1,200 段则多次更换 seed）
python scripts/04_generate.py --mode hf --model-id Qwen/Qwen2.5-0.5B-Instruct \
       --deltas 0,0.5,1,2 --max-new-tokens 200 \
       --vocab 151936 --out outputs/hf_qwen.jsonl --fresh
```

## 6. 把结果带回来

训练/推理实例通常是临时的，**先把产物拷回本地**：

```bash
# 在本地执行（把用户名/主机/端口换成实例信息）
scp -P <端口> root@<实例IP>:/root/WatermarkSys_LLMText/WatermarkSys_LLMText/outputs/hf_qwen* ./outputs/
```

然后本地重跑评估与报告（不需要 GPU）：

```bash
python scripts/08_eval.py --detections outputs/hf_qwen_detections.jsonl
python scripts/10_report.py
```

## 7. 常见坑

| 现象 | 原因 | 处理 |
|---|---|---|
| `--vocab` 用了 2048 | 默认值是给离线模拟器用的 | 真实模型必须传真实词表大小 |
| `cuda.is_available()` 为 False | 装了 CPU 版 torch | 重装 cu121 版本 |
| 显存 OOM | 一次加载多个模型 / 长度过大 | 降 `--max-new-tokens`，换个实例 |
| 下载中断 | 带宽限速 | 用 `snapshot_download` 可断点续传，重跑同一命令即可 |
| 生成很慢但 GPU 空闲 | 模型仍在 CPU | 检查 `device` 与 torch 版本 |
| 结果 z 异常低 | 生成/检测 vocab、gamma、h 不一致 | 核对 `config_hash` 与 `--vocab` |

## 8. 时间与预算估算

| 方案 | 时间成本 | 金钱成本 | 适用 |
|---|---|---|---|
| 学校 GPU | 申请 0.5—1 天 | 0 元 | 有实验室资源 |
| AI Studio / 魔搭 | 注册 10 分钟 | 0 元 | 想零成本 |
| AutoDL 3090 | 开机 10 分钟 | 约 3—5 元 | 今天就要结果 |

> 如果暂时拿不到 GPU：现有离线预演 + tiny-gpt2 真实权重验证已足以支撑论文的
> “机制正确、口径严谨、可复现”三部分；主模型数字作为有条件时的补强。
