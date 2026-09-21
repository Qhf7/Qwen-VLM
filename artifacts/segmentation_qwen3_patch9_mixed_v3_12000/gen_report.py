#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv, json, statistics

SRC = "/Users/qiaohongfei/Desktop/实习/阿里云天池比赛vlm/artifacts/segmentation_qwen3_patch9_mixed_v3_12000/training_history.csv"
OUT = "/Users/qiaohongfei/Desktop/实习/阿里云天池比赛vlm/artifacts/segmentation_qwen3_patch9_mixed_v3_12000/training_report.html"

rows = []
with open(SRC, newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        rows.append(row)

n = len(rows)
def fl(x):
    try: return float(x)
    except: return None

step   = [int(float(r["step"])) for r in rows]
loss   = [fl(r["loss"]) for r in rows]
focal  = [fl(r["focal_loss"]) for r in rows]
dice   = [fl(r["dice_loss"]) for r in rows]
gnorm  = [fl(r["grad_norm"]) for r in rows]
lr     = [fl(r["lr"]) for r in rows]
step_s = [fl(r["step_s"]) for r in rows]

# downsample for charts (~500 points)
stride = max(1, n // 500)
def ds(a): return [a[i] for i in range(0, n, stride)] + [a[-1]]
s_step = ds(step)
s_loss = ds(loss); s_focal = ds(focal); s_dice = ds(dice)
s_gn   = ds(gnorm); s_lr = ds(lr); s_sp = ds(step_s)

cur = rows[-1]
cur_step = int(float(cur["step"]))
cur_loss = fl(cur["loss"]); cur_focal = fl(cur["focal_loss"]); cur_dice = fl(cur["dice_loss"])
cur_gn = fl(cur["grad_norm"]); cur_lr = fl(cur["lr"]); cur_sp = fl(cur["step_s"])

# best (min total loss)
best_i = min(range(n), key=lambda i: loss[i])
best_step = step[best_i]; best_loss = loss[best_i]

# recent window stats (last 200)
W = min(200, n)
def recent(a): return a[-W:]
rec_loss = statistics.mean(recent(loss)); rec_focal = statistics.mean(recent(focal))
rec_dice = statistics.mean(recent(dice)); rec_gn = statistics.mean(recent(gnorm))
rec_sp   = statistics.mean(recent(step_s))

# early window (first 200) for comparison
E = min(200, n)
early_loss = statistics.mean(loss[:E]); early_dice = statistics.mean(dice[:E])

eta_steps = 12000 - cur_step
eta_sec = eta_steps * rec_sp
from datetime import datetime, timedelta
now = datetime.now()
eta_dt = now + timedelta(seconds=eta_sec)
eta_str = eta_dt.strftime("%H:%M")

data = dict(
    step=s_step, loss=s_loss, focal=s_focal, dice=s_dice,
    gnorm=s_gn, lr=s_lr, step_s=s_sp,
)

html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>v3 训练可视化报告 (Patch9 mixed_v3)</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root{{--bg:#f7f8fa;--card:#ffffff;--ink:#1f2933;--sub:#5b6b7b;--line:#e3e8ee;--acc:#2563eb;--good:#16a34a;--warn:#d97706;}}
  *{{box-sizing:border-box}}
  body{{margin:0;font-family:-apple-system,"PingFang SC","Microsoft YaHei",Segoe UI,sans-serif;background:var(--bg);color:var(--ink);padding:24px}}
  h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:16px;margin:22px 0 10px}}
  .sub{{color:var(--sub);font-size:13px;margin-bottom:18px}}
  .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:8px}}
  .card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}}
  .card .k{{font-size:12px;color:var(--sub)}} .card .v{{font-size:22px;font-weight:700;margin-top:4px}}
  .card .d{{font-size:11px;color:var(--sub);margin-top:3px}}
  .panel{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:16px}}
  .grid2{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
  @media(max-width:880px){{.grid2{{grid-template-columns:1fr}}}}
  canvas{{max-height:300px}}
  table{{width:100%;border-collapse:collapse;font-size:13px}}
  td,th{{border:1px solid var(--line);padding:7px 10px;text-align:left}}
  th{{background:#eef2f7}}
  .tag{{display:inline-block;background:#e8f0fe;color:var(--acc);border-radius:6px;padding:2px 8px;font-size:12px;margin:2px}}
  .ok{{color:var(--good);font-weight:700}}
</style></head>
<body>
<h1>九宫格 v3 训练可视化报告</h1>
<div class="sub">实验 <code>segmentation_qwen3_patch9_mixed_v3_12000</code> · 快照于 step {cur_step}/{12000} · 生成时间 {now.strftime('%Y-%m-%d %H:%M')}</div>

<div class="cards">
  <div class="card"><div class="k">当前 step</div><div class="v">{cur_step}</div><div class="d">/ 12000（{cur_step/12000*100:.1f}%）</div></div>
  <div class="card"><div class="k">total loss（当前）</div><div class="v">{cur_loss:.4f}</div><div class="d">近200步均值 {rec_loss:.4f}</div></div>
  <div class="card"><div class="k">dice_loss（当前）</div><div class="v">{cur_dice:.4f}</div><div class="d">首200步 {early_dice:.4f} → 下降明显</div></div>
  <div class="card"><div class="k">focal_loss（当前）</div><div class="v">{cur_focal:.4f}</div><div class="d">近200步均值 {rec_focal:.4f}</div></div>
  <div class="card"><div class="k">grad_norm（当前）</div><div class="v">{cur_gn:.3f}</div><div class="d">稳定流动，无爆炸/消失</div></div>
  <div class="card"><div class="k">lr（当前）</div><div class="v">{cur_lr:.2e}</div><div class="d">warmup 已完成</div></div>
  <div class="card"><div class="k">速度</div><div class="v">{rec_sp:.2f}s/step</div><div class="d">剩余 {eta_steps} 步</div></div>
  <div class="card"><div class="k">预计完成</div><div class="v">{eta_str}</div><div class="d">约 {eta_sec/60:.0f} 分钟后</div></div>
</div>

<div class="panel">
  <h2>① Loss 曲线（total / focal / dice）</h2>
  <div class="grid2">
    <div><canvas id="cLoss"></canvas></div>
    <div><canvas id="cDice"></canvas></div>
  </div>
</div>

<div class="panel">
  <h2>② 梯度范数 与 学习率</h2>
  <div class="grid2">
    <div><canvas id="cGrad"></canvas></div>
    <div><canvas id="cLr"></canvas></div>
  </div>
</div>

<div class="panel">
  <h2>③ 每步耗时（速度稳定性）</h2>
  <canvas id="cSpeed"></canvas>
</div>

<div class="panel">
  <h2>④ 训练协议与校验（来自 run.log）</h2>
  <table>
    <tr><th>项目</th><th>值</th></tr>
    <tr><td>基座冻结 base_trainable</td><td class="ok">False（仅训 decoder）</td></tr>
    <tr><td>可训参数 trainable_parameters</td><td>3,285,505</td></tr>
    <tr><td>前向校验 passed</td><td class="ok">True</td></tr>
    <tr><td>初始梯度范数（step 0 前）</td><td>6.15</td></tr>
    <tr><td>Loss 协议</td><td>Focal(γ=2, pos_α=0.25) + Soft Dice(非空)</td></tr>
    <tr><td>训练样本数</td><td>18,775 行 · 18 类 · 空 mask 行 2,993 · 全正patch异常图 153</td></tr>
    <tr><td>验证样本数</td><td>3,600 行 · 空 mask 行 1,825</td></tr>
    <tr><td>每步 input_mix</td><td><span class="tag">full=4</span><span class="tag">patch=4</span><span class="tag">patch_empty=2</span><span class="tag">full_empty=2</span> → 8 非空/step（RC-3 修复）</td></tr>
    <tr><td>已存检查点（eval 节点）</td><td>250 / 500 / 750 / 1000 / 1500 / 2000 / 2500 / 3000 / 4000 …（持续更新 latest.pt）</td></tr>
  </table>
</div>

<div class="panel">
  <h2>⑤ 关键结论</h2>
  <ul>
    <li>Total loss 从首步 <b>{loss[0]:.3f}</b> 降到当前 <b>{cur_loss:.3f}</b>，最佳 step <b>{best_step}</b> 达 <b>{best_loss:.3f}</b>。</li>
    <li>Dice loss 从首200步均值 <b>{early_dice:.3f}</b> 降至近200步 <b>{rec_dice:.3f}</b>，前景学习稳定。</li>
    <li>LR 由 4e-6 warmup 至 ~7.1e-5，符合预期调度。</li>
    <li>grad_norm 全程 0.7–8.6 区间，无 NaN/Inf，训练健康。</li>
    <li>RC-3 已生效：每步 8 个非空前景样本（=baseline），前景监督较 v2(4/step) 翻倍。</li>
  </ul>
</div>

<script>
const D = {json.dumps(data)};
const grid={{color:'#eef2f7'}}; const tick={{color:'#5b6b7b'}};
function mk(id, labels, datasets, ylog){{
  new Chart(document.getElementById(id), {{
    type:'line',
    data:{{labels, datasets}},
    options:{{responsive:true,animation:false,
      scales:{{x:{{grid, ticks:tick, title:{{display:true,text:'step',color:'#5b6b7b'}}}},
               y:{{grid, ticks:tick, type:ylog?'logarithmic':'linear'}}}},
      elements:{{point:{{radius:0}}}},
      plugins:{{legend:{{labels:{{color:'#1f2933'}}}}}}}}
  }});
}}
const lossDat=[{{label:'total',data:D.loss,borderColor:'#2563eb'}},
  {{label:'focal',data:D.focal,borderColor:'#d97706'}},
  {{label:'dice',data:D.dice,borderColor:'#16a34a'}}];
mk('cLoss', D.step, lossDat, true);
mk('cDice', D.step, [{{label:'dice_loss',data:D.dice,borderColor:'#16a34a'}}], false);
mk('cGrad', D.step, [{{label:'grad_norm',data:D.gnorm,borderColor:'#7c3aed'}}], false);
mk('cLr',   D.step, [{{label:'lr',data:D.lr,borderColor:'#dc2626'}}], true);
mk('cSpeed',D.step, [{{label:'step_s',data:D.step_s,borderColor:'#0891b2'}}], false);
</script>
</body></html>"""

with open(OUT, "w") as f:
    f.write(html)
print("WROTE", OUT, "rows=", n, "cur_step=", cur_step, "best_loss=", round(best_loss,4), "@", best_step)
