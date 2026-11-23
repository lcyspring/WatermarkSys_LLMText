"""Web 演示系统（W10 交付物）。

提供最小可用的可视化界面与 JSON API：
  GET  /              仪表盘（读取已生成指标）
  GET  /api/health    健康检查
  GET  /api/config    当前配置快照
  GET  /api/metrics   指标摘要
  POST /api/generate  按提示生成文本（默认离线机制模拟）
  POST /api/detect    对给定文本/token 做水印检测
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

PACKAGE_SRC = Path(__file__).resolve().parents[2]   # .../WatermarkSys_LLMText/src
CODE_ROOT = PACKAGE_SRC.parent                      # .../WatermarkSys_LLMText
sys.path.insert(0, str(PACKAGE_SRC))

from wm.config import WatermarkConfig  # noqa: E402
from wm.model_adapter import SyntheticLM  # noqa: E402
from wm.watermark import detect as kgw_detect  # noqa: E402

OUT = CODE_ROOT / "outputs"
DEV_KEY = bytes.fromhex(
    "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f")

app = FastAPI(title="LLM 文本水印检测演示", version="0.2.0")


class GenerateReq(BaseModel):
    prompt: str
    delta: float = 2.0
    max_new_tokens: int = 120
    vocab_size: int = 2048
    seed: int = 0


class DetectReq(BaseModel):
    text: str = ""
    token_ids: list[int] | None = None
    delta: float = 2.0
    vocab_size: int = 2048
    gamma: float = 0.25
    threshold: float | None = None


def _load_metrics() -> dict:
    p = OUT / "metrics_summary.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def _default_threshold() -> float | None:
    p = OUT / "threshold_v1.json"
    if p.exists():
        return float(json.loads(p.read_text(encoding="utf-8"))["threshold"])
    return None


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": app.version}


@app.get("/api/config")
def get_config() -> dict:
    cfg = WatermarkConfig(vocab_size=2048, delta=2.0)
    return {"config": cfg.to_dict(), "config_hash": cfg.config_hash()}


@app.get("/api/metrics")
def metrics() -> dict:
    return _load_metrics()


@app.post("/api/generate")
def generate(req: GenerateReq) -> dict:
    cfg = WatermarkConfig(delta=req.delta, vocab_size=req.vocab_size,
                          max_new_tokens=req.max_new_tokens, seed=req.seed)
    lm = SyntheticLM(vocab_size=req.vocab_size)
    prompt_ids = lm.encode(req.prompt)
    new_ids = lm.generate_ids(prompt_ids, cfg, DEV_KEY,
                              max_new_tokens=req.max_new_tokens)
    ids = prompt_ids + new_ids
    det = kgw_detect(ids, key=DEV_KEY, gamma=cfg.gamma,
                     context_width=cfg.context_width,
                     vocab_size=cfg.vocab_size,
                     ngram_block=cfg.ngram_block,
                     threshold=_default_threshold(),
                     score_from=len(prompt_ids))
    return {
        "text": lm.decode(new_ids),
        "prompt_len": len(prompt_ids),
        "n_new_tokens": len(new_ids),
        "config_hash": cfg.config_hash(),
        "detection": {k: det[k] for k in
                      ("status", "n", "g", "z", "p", "decision", "threshold")},
    }


@app.post("/api/detect")
def detect(req: DetectReq) -> dict:
    lm = SyntheticLM(vocab_size=req.vocab_size)
    ids = req.token_ids if req.token_ids is not None else lm.encode(req.text)
    tau = req.threshold if req.threshold is not None else _default_threshold()
    res = kgw_detect(ids, key=DEV_KEY, gamma=req.gamma,
                     context_width=4, vocab_size=req.vocab_size,
                     threshold=tau, score_from=4)
    return {k: res[k] for k in
            ("status", "n", "g", "z", "p", "decision", "threshold",
             "min_tokens")}


PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>LLM 文本水印检测演示</title>
<style>
:root{--navy:#0F2A43;--blue:#1B6CA8;--teal:#14B8A6;--bg:#F5F7FA;}
*{box-sizing:border-box;}body{margin:0;font-family:"Microsoft YaHei",sans-serif;
background:var(--bg);color:#1f2937;}
header{background:var(--navy);color:#fff;padding:18px 28px;}
header h1{margin:0;font-size:20px;font-weight:600;}
header p{margin:6px 0 0;font-size:13px;opacity:.8;}
.wrap{max-width:1080px;margin:24px auto;padding:0 20px;}
.card{background:#fff;border-radius:10px;padding:20px;margin-bottom:18px;
box-shadow:0 1px 3px rgba(15,42,67,.10);}
h2{font-size:15px;margin:0 0 14px;color:var(--navy);}
textarea{width:100%;height:88px;padding:10px;border:1px solid #d1d5db;
border-radius:8px;font-family:inherit;font-size:14px;resize:vertical;}
.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-top:12px;}
button{background:var(--blue);color:#fff;border:0;border-radius:8px;
padding:10px 18px;font-size:14px;cursor:pointer;}
button:hover{background:#155a8f;}
input[type=number]{width:90px;padding:8px;border:1px solid #d1d5db;border-radius:8px;}
pre{background:#0F2A43;color:#c9e4ff;padding:14px;border-radius:8px;
overflow:auto;font-size:12.5px;line-height:1.6;}
.badge{display:inline-block;padding:3px 10px;border-radius:999px;font-size:12px;}
.ok{background:#dcfce7;color:#166534;}.no{background:#fee2e2;color:#991b1b;}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;}
.stat{background:#F8FAFC;border:1px solid #e5e7eb;border-radius:8px;padding:12px;}
.stat b{display:block;font-size:20px;color:var(--navy);}
.stat span{font-size:12px;color:#6b7280;}
</style></head><body>
<header><h1>面向大语言模型生成文本的水印嵌入与鲁棒检测系统</h1>
<p>KGW 绿名单水印 · 演示环境（离线机制模拟）· 检测结论仅表示“是否检出指定配置的水印证据”</p></header>
<div class="wrap">
<div class="card"><h2>指标概览</h2><div class="grid" id="stats">
<div class="stat"><b>--</b><span>加载中</span></div></div></div>
<div class="card"><h2>生成并检测</h2>
<textarea id="prompt">请用一段话说明大语言模型文本水印的基本原理。</textarea>
<div class="row">
<label>delta <input type="number" id="delta" value="2" step="0.5" min="0"></label>
<label>生成长度 <input type="number" id="maxtok" value="120" step="20"></label>
<button onclick="run()">生成并检测</button>
<span id="verdict"></span></div>
<pre id="out">点击“生成并检测”开始。</pre></div>
<div class="card"><h2>说明</h2>
<pre id="hint" style="background:#F8FAFC;color:#374151;border:1px solid #e5e7eb;">
1. delta=0 表示完全不偏置（无水印基线），detection.z 应在 0 附近。
2. delta 越大，绿名单命中率越高，z 与 p 越显著。
3. 判定阈值由负例校准得到，目标 FPR=1%。
4. 未检出不等于文本由人类撰写；检出也不代表内容真实。</pre></div>
</div>
<script>
async function loadMetrics(){
  try{const r=await fetch('/api/metrics');const m=await r.json();
    const el=document.getElementById('stats');el.innerHTML='';
    if(!m.by_delta){el.innerHTML='<div class="stat"><b>--</b><span>尚无指标，请先运行实验脚本</span></div>';return;}
    const add=(t,v)=>{const d=document.createElement('div');d.className='stat';
      d.innerHTML='<b>'+v+'</b><span>'+t+'</span>';el.appendChild(d);};
    add('阈值 z', m.threshold);
    Object.keys(m.by_delta).forEach(k=>{const b=m.by_delta[k];
      add('delta='+k+' 平均 z', b.z_mean);
      add('delta='+k+' 检出率', (100*b.detected_rate).toFixed(1)+'%');});
  }catch(e){}
}
async function run(){
  const prompt=document.getElementById('prompt').value;
  const delta=parseFloat(document.getElementById('delta').value);
  const maxtok=parseInt(document.getElementById('maxtok').value);
  document.getElementById('out').textContent='生成中…';
  const r=await fetch('/api/generate',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({prompt:prompt,delta:delta,max_new_tokens:maxtok})});
  const d=await r.json();
  document.getElementById('out').textContent=
    '生成文本（截断显示）：\\n'+d.text.slice(0,400)+
    '\\n\\n检测结果：\\n'+JSON.stringify(d.detection,null,2);
  const det=d.detection;
  const v=document.getElementById('verdict');
  v.innerHTML = det.decision===true?'<span class="badge ok">检出本方案水印</span>'
    : det.decision===false?'<span class="badge no">未检出</span>'
    : '<span class="badge no">有效长度不足</span>';
}
loadMetrics();
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE
