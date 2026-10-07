#!/usr/bin/env python
"""Summarise scripts/eval_hf_models.sh's evaluations: report.md, figures, an interactive curves.html.

    python scripts/eval_hf_report.py --out /data/ironman/jiacheng/rdm-eval

Reads <out>/models.json and every <out>/evals/**/summary.json, and writes into <out>:

    report.md       table 1: each run's best checkpoint (by GenEval) next to the two baselines;
                    the best checkpoint whose seed diversity stays >= 0.9x the teacher's; GenEval
                    per task; the three figures; every checkpoint's numbers
    fig_*.png       GenEval / PickScore / seed diversity (DreamSim) against the training step
    curves.html     the same curves, interactive: pick the metric, tick any runs and baselines
    results.csv     one row per evaluation

Safe to run at any time -- it reports whatever is finished.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import time

TASKS = ["single_object", "two_object", "counting", "colors", "position", "color_attr"]
TASK_ZH = ["单物体", "两物体", "计数", "颜色", "位置", "属性绑定"]
METRICS = [
    {"key": "geneval", "label": "GenEval", "digits": 4,
     "hint": "553 条 prompt × 4 个种子，6 类任务正确率的平均，越高越好"},
    {"key": "pickscore", "label": "PickScore", "digits": 3,
     "hint": "Pick-a-Pic 499 条 prompt 每条 1 张，越高越好"},
    {"key": "div_dreamsim", "label": "种子多样性 · DreamSim", "digits": 4,
     "hint": "同一 prompt 4 张图两两 1 − 余弦相似度的平均，越高越多样，越低越接近 mode collapse"},
    {"key": "div_pixel", "label": "种子多样性 · pixel", "digits": 4,
     "hint": "同一 prompt 4 张图 64 px 灰度的平均绝对差，反映构图差异"},
    {"key": "div_dinov3", "label": "种子多样性 · DINOv3", "digits": 4,
     "hint": "同一 prompt 4 张图 DINOv3 特征两两 1 − 余弦相似度的平均，反映语义差异"},
]
# categorical slots 1-7 of the validated reference palette (light / dark), in fixed order
LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9"]
MARKERS = ["o", "s", "D", "^", "v", "p", "X"]


def record(path: str) -> dict | None:
    try:
        s = json.load(open(path))
    except (OSError, ValueError):
        return None
    g = s.get("geneval") or {}
    d = s.get("seed_diversity") or {}
    return {"geneval": g.get("overall"), "tasks": {t: g.get(t) for t in TASKS},
            "pickscore": s.get("pickscore"), "div_pixel": d.get("pixel"),
            "div_dreamsim": d.get("dreamsim"), "div_dinov3": d.get("dinov3_l")}


def collect(out: str):
    meta = json.load(open(os.path.join(out, "models.json")))
    runs = []
    for m in meta["runs"]:
        points = []
        for path in sorted(glob.glob(os.path.join(out, "evals", m["key"], "step_*", "summary.json"))):
            rec = record(path)
            if rec:
                rec["step"] = int(re.search(r"step_(\d+)", path).group(1))
                points.append(rec)
        runs.append({**m, "points": sorted(points, key=lambda p: p["step"])})
    baselines = []
    for b in meta.get("baselines", []):
        rec = record(os.path.join(out, "evals", "baselines", b["key"], "summary.json"))
        baselines.append({**b, "values": rec})
    return meta, runs, baselines


def best(points, key="geneval", where=None):
    cands = [p for p in points if p.get(key) is not None and (where is None or where(p))]
    return max(cands, key=lambda p: p[key]) if cands else None


def f(v, digits):
    return "–" if v is None else f"{v:.{digits}f}"


def gpu_name() -> str:
    try:
        import subprocess
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=30).stdout.splitlines()[0].strip()
    except Exception:
        return "?"


def figure(path, runs, baselines, key, ylabel, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11.5, 5.0), dpi=110)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    drawn = False
    for i, r in enumerate(runs):
        pts = [(p["step"], p[key]) for p in r["points"] if p.get(key) is not None]
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=LIGHT[i % len(LIGHT)], lw=2, marker=MARKERS[i % len(MARKERS)],
                    ms=6.5, mec="#fcfcfb", mew=1.2, label=r["label"])
            drawn = True
    for (ls, color), b in zip([("--", "#52514e"), (":", "#0b0b0b")], baselines):
        v = (b["values"] or {}).get(key)
        if v is not None:
            ax.axhline(v, ls=ls, color=color, lw=1.6, label=f"{b['label']}: {v:.4g}")
            drawn = True
    if not drawn:
        plt.close(fig)
        return False
    ax.set_title(title, loc="left", fontsize=12, color="#0b0b0b")
    ax.set_xlabel("training step", color="#52514e")
    ax.set_ylabel(ylabel, color="#52514e")
    ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c1")
    ax.tick_params(colors="#52514e")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


def write_markdown(out, meta, runs, baselines, gpu):
    teacher = next((b["values"] for b in baselines if b["key"] == "teacher_4step" and b["values"]), None)
    t_div = (teacher or {}).get("div_dreamsim")

    def ratio(v):
        return "" if v is None or not t_div else f"（{v / t_div:.2f}×）"

    n_done = sum(len(r["points"]) for r in runs) + sum(1 for b in baselines if b["values"])
    n_all = sum(len(r.get("steps", [])) for r in runs) + len(baselines)
    L = ["# SW-LMMD 模型评测汇总", "",
         f"生成于 {time.strftime('%Y-%m-%d %H:%M')} · 评测 GPU：{gpu} · 已完成 {n_done} / {n_all} 个评测", "",
         "## 评测协议", "",
         "- **GenEval**：官方 553 条 prompt × 每条 4 张（噪声种子 46,000,000 + prompt 编号 × 100 + j），"
         "Mask2Former 检测 + CLIP 判断颜色；总分是 6 类任务正确率的平均。",
         "- **PickScore**：Pick-a-Pic 测试集 499 条 prompt，每条 1 张，文本长度 232。",
         "- **种子多样性**：同一条 GenEval prompt 的 4 张图两两比较（6 对）取平均，再对 553 条 prompt 平均。"
         "**DreamSim**、**DINOv3** 是 1 − 余弦相似度，**pixel** 是 64 px 灰度的平均绝对差。越高越多样，"
         "明显低于老师就是 mode collapse 的迹象。",
         "- 学生都是 1 步采样，老师（klein-4B）4 步。所有数字都在同一台机器、同一套流程下测得。", ""]

    # ---- table 1
    cols = "| 模型 | 设置 | 最好的 step | GenEval | PickScore | 多样性 DreamSim（×老师） | 多样性 pixel | 多样性 DINOv3 | 已评测 |"
    rows = []
    for r in runs:
        b = best(r["points"])
        n = f"{len(r['points'])}/{len(r.get('steps', []))}"
        if b is None:
            rows.append([r["label"], r["desc"], "–", None, None, None, None, None, n])
        else:
            rows.append([r["label"], r["desc"], str(b["step"]), b["geneval"], b["pickscore"],
                         b["div_dreamsim"], b["div_pixel"], b["div_dinov3"], n])
    for b in baselines:
        v = b["values"] or {}
        rows.append([b["label"], b["desc"], "–", v.get("geneval"), v.get("pickscore"),
                     v.get("div_dreamsim"), v.get("div_pixel"), v.get("div_dinov3"),
                     "1/1" if b["values"] else "0/1"])
    colmax = {i: max((row[i] for row in rows if row[i] is not None), default=None) for i in range(3, 8)}
    digits = {3: 4, 4: 3, 5: 4, 6: 4, 7: 4}

    def cell(row, i):
        txt = f(row[i], digits[i]) + (ratio(row[i]) if i == 5 else "")
        return f"**{txt}**" if row[i] is not None and row[i] == colmax[i] else txt

    L += ["## 表 1：各模型最好的 checkpoint", "",
          "每个模型在所有 checkpoint 里按 GenEval 选最高的那个；加粗是每列最高。", "",
          cols, "|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        L.append(f"| {row[0]} | {row[1]} | {row[2]} | " + " | ".join(cell(row, i) for i in range(3, 8))
                 + f" | {row[8]} |")
    L += ["", "> 在 10–15 个 checkpoint 里挑最高的，用的就是测试集本身，所以会比随便取一个点偏高约 0.01"
              "（相邻 checkpoint 之间的起伏）；各模型都按同样方法挑，彼此可比。", ""]

    # ---- table 2: best under the diversity constraint
    if t_div:
        L += ["## 表 2：多样性不低于老师 0.9 倍时最好的 checkpoint", "",
              f"约束：DreamSim 多样性 ≥ 0.9 × 老师（{t_div:.4f}）= {0.9 * t_div:.4f}。"
              "这是同时要求“不 mode collapse”时能拿到的最好成绩。", "",
              "| 模型 | step | GenEval | PickScore | 多样性 DreamSim（×老师） | 满足约束的 checkpoint |",
              "|---|---|---|---|---|---|"]
        for r in runs:
            ok = [p for p in r["points"] if p.get("div_dreamsim") is not None and p["div_dreamsim"] >= 0.9 * t_div]
            b = best(ok)
            if b is None:
                L.append(f"| {r['label']} | – | – | – | – | 0/{len(r['points'])} |")
            else:
                L.append(f"| {r['label']} | {b['step']} | {f(b['geneval'], 4)} | {f(b['pickscore'], 3)} | "
                         f"{f(b['div_dreamsim'], 4)}{ratio(b['div_dreamsim'])} | {len(ok)}/{len(r['points'])} |")
        L.append("")

    # ---- table 3: GenEval per task at the best checkpoint
    L += ["## 表 3：最好 checkpoint 的 GenEval 分项（通过率 %）", "",
          "| 模型 | step | " + " | ".join(TASK_ZH) + " | 总分 |", "|---|---|" + "---|" * (len(TASKS) + 1)]
    for r in runs:
        b = best(r["points"])
        if b:
            L.append(f"| {r['label']} | {b['step']} | " + " | ".join(
                f(None if b["tasks"][t] is None else 100 * b["tasks"][t], 1) for t in TASKS)
                + f" | {f(b['geneval'], 4)} |")
    for b in baselines:
        v = b["values"]
        if v:
            L.append(f"| {b['label']} | – | " + " | ".join(
                f(None if v["tasks"][t] is None else 100 * v["tasks"][t], 1) for t in TASKS)
                + f" | {f(v['geneval'], 4)} |")
    L.append("")

    # ---- figures
    L += ["## 图：随训练步数的变化", "",
          "**交互版：[curves.html](curves.html)**（用浏览器打开）——可以切换纵轴指标（GenEval / PickScore / "
          "三种多样性），并勾选任意几个模型和两个基准单独比较。下面是三个主要指标的静态图（全部模型）。", ""]
    for name, title in [("fig_geneval.png", "GenEval"), ("fig_pickscore.png", "PickScore"),
                        ("fig_div_dreamsim.png", "种子多样性（DreamSim）")]:
        if os.path.isfile(os.path.join(out, name)):
            L += [f"### {title}", "", f"![{title}]({name})", ""]

    # ---- appendix: every checkpoint
    L += ["## 附：每个 checkpoint 的完整结果", ""]
    for r in runs:
        L += [f"<details><summary><b>{r['label']}</b> — {r['desc']}（{r['repo']}）</summary>", "",
              "| step | GenEval | " + " | ".join(TASK_ZH) + " | PickScore | DreamSim | pixel | DINOv3 |",
              "|---|---|" + "---|" * (len(TASKS) + 4)]
        for p in r["points"]:
            L.append(f"| {p['step']} | {f(p['geneval'], 4)} | " + " | ".join(
                f(None if p["tasks"][t] is None else 100 * p["tasks"][t], 1) for t in TASKS)
                + f" | {f(p['pickscore'], 3)} | {f(p['div_dreamsim'], 4)} | {f(p['div_pixel'], 4)} | "
                  f"{f(p['div_dinov3'], 4)} |")
        L += ["", "</details>", ""]

    missing = [f"{r['key']}/step_{s:07d}" for r in runs for s in r.get("steps", [])
               if s not in {p["step"] for p in r["points"]}]
    missing += [f"baselines/{b['key']}" for b in baselines if not b["values"]]
    if missing:
        L += ["## 尚未完成的评测", "", ", ".join(f"`{m}`" for m in missing), ""]
    open(os.path.join(out, "report.md"), "w").write("\n".join(L))


def write_csv(out, runs, baselines):
    with open(os.path.join(out, "results.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "label", "step", "geneval", *TASKS, "pickscore",
                    "div_dreamsim", "div_pixel", "div_dinov3"])
        for r in runs:
            for p in r["points"]:
                w.writerow([r["key"], r["label"], p["step"], p["geneval"], *[p["tasks"][t] for t in TASKS],
                            p["pickscore"], p["div_dreamsim"], p["div_pixel"], p["div_dinov3"]])
        for b in baselines:
            v = b["values"]
            if v:
                w.writerow([b["key"], b["label"], "", v["geneval"], *[v["tasks"][t] for t in TASKS],
                            v["pickscore"], v["div_dreamsim"], v["div_pixel"], v["div_dinov3"]])


def write_html(out, runs, baselines, gpu):
    keys = [m["key"] for m in METRICS]
    data = {
        "created": time.strftime("%Y-%m-%d %H:%M"), "gpu": gpu, "metrics": METRICS,
        "runs": [{"key": r["key"], "label": r["label"], "desc": r["desc"],
                  "points": [{"step": p["step"], **{k: p.get(k) for k in keys}} for p in r["points"]]}
                 for r in runs],
        "baselines": [{"key": b["key"], "label": b["label"], "desc": b["desc"],
                       "values": {k: (b["values"] or {}).get(k) for k in keys}} for b in baselines],
    }
    page = HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    open(os.path.join(out, "curves.html"), "w").write(page)


HTML = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SW-LMMD 评测曲线</title>
<style>
:root {
  color-scheme: light;
  --bg: #fcfcfb; --surface: #ffffff; --ink: #0b0b0b; --muted: #52514e; --line: #deddd7;
  --grid: #ecebe6; --axis: #c9c8c1; --focus: #2a78d6;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #e87ba4; --s6: #008300; --s7: #4a3aa7;
  --b1: #52514e; --b2: #0b0b0b;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --bg: #1a1a19; --surface: #222220; --ink: #ffffff; --muted: #c3c2b7; --line: #3a3a37;
    --grid: #2d2d2a; --axis: #4a4a46; --focus: #3987e5;
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #008300; --s7: #9085e9;
    --b1: #c3c2b7; --b2: #ffffff;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg: #1a1a19; --surface: #222220; --ink: #ffffff; --muted: #c3c2b7; --line: #3a3a37;
  --grid: #2d2d2a; --axis: #4a4a46; --focus: #3987e5;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #008300; --s7: #9085e9;
  --b1: #c3c2b7; --b2: #ffffff;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", "PingFang SC", "Noto Sans SC", "Microsoft YaHei", sans-serif; }
.wrap { max-width: 1120px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 22px; margin: 0 0 4px; }
.sub { color: var(--muted); margin: 0 0 18px; font-size: 13px; }
.row { display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: center; margin-bottom: 12px; }
.row > .lab { color: var(--muted); font-size: 13px; min-width: 3em; }
.seg { display: inline-flex; flex-wrap: wrap; gap: 6px; }
button { font: inherit; color: var(--ink); background: var(--surface); border: 1px solid var(--line);
  border-radius: 999px; padding: 4px 12px; cursor: pointer; }
button[aria-pressed="true"] { background: var(--ink); color: var(--bg); border-color: var(--ink); }
button:focus-visible, input:focus-visible, svg:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }
.series { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 4px 16px; flex: 1 1 600px; }
.series label { display: flex; align-items: center; gap: 8px; cursor: pointer; padding: 2px 0; min-width: 0; }
.series label span.t { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.series svg { flex: none; }
.card { margin: 8px 0 0; background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 12px 12px 6px; }
.card figcaption { display: flex; flex-wrap: wrap; gap: 2px 10px; align-items: baseline; }
.card figcaption span { color: var(--muted); font-size: 12px; }
.plot { position: relative; }
#svg { display: block; width: 100%; }
.tick { font-size: 11px; fill: var(--muted); font-variant-numeric: tabular-nums; }
.axl { font-size: 12px; fill: var(--muted); }
.gridln { stroke: var(--grid); stroke-width: 1; }
.axisln { stroke: var(--axis); stroke-width: 1; }
.ln { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
path.mk { stroke: var(--surface); stroke-width: 1.5; }   /* the 2px surface ring, over .sN */
.dl { font-size: 12px; fill: var(--ink); }
.dlv { font-size: 11px; fill: var(--muted); font-variant-numeric: tabular-nums; }
.base { stroke-width: 1.6; fill: none; }
.cross { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 3 3; }
.empty { font-size: 13px; fill: var(--muted); }
""" + "".join(f".s{i + 1} {{ stroke: var(--s{i + 1}); fill: var(--s{i + 1}); }}\n" for i in range(7)) + r"""
.ln.s1, .ln.s2, .ln.s3, .ln.s4, .ln.s5, .ln.s6, .ln.s7 { fill: none; }
.b1 { stroke: var(--b1); } .b2 { stroke: var(--b2); }
.tip { position: absolute; top: 8px; pointer-events: none; background: var(--surface); border: 1px solid var(--line);
  border-radius: 6px; padding: 6px 9px; font-size: 12px; min-width: 190px; box-shadow: 0 6px 18px rgba(0,0,0,.14); }
.tip .h { color: var(--muted); margin-bottom: 2px; }
.tip .r { display: flex; align-items: center; gap: 7px; white-space: nowrap; }
.tip .r b { font-variant-numeric: tabular-nums; min-width: 4.4em; }
.tip .r span { color: var(--muted); }
details { margin-top: 14px; }
summary { cursor: pointer; color: var(--focus); width: max-content; }
.scroll { overflow-x: auto; margin-top: 8px; border: 1px solid var(--line); border-radius: 6px; background: var(--surface); }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; font-size: 13px; }
th, td { padding: 6px 10px; text-align: right; border-bottom: 1px solid var(--line); white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
th { color: var(--muted); font-weight: 500; }
.note { color: var(--muted); font-size: 12px; margin-top: 12px; max-width: 90ch; }
</style>
</head>
<body>
<div class="wrap">
  <h1>SW-LMMD 评测：随训练步数的变化</h1>
  <p class="sub" id="sub"></p>
  <div class="row"><span class="lab">纵轴</span><div class="seg" id="metrics" role="group" aria-label="指标"></div>
    <span style="flex:1"></span><button id="theme" type="button" title="切换浅色 / 深色">主题：自动</button></div>
  <div class="row"><span class="lab">显示</span><div class="series" id="series"></div></div>
  <div class="row"><span class="lab"></span><div class="seg">
    <button id="all" type="button">全选</button><button id="none" type="button">全不选</button>
    <button id="onlyb" type="button">只看基准</button></div></div>
  <figure class="card">
    <figcaption><b id="mtitle"></b><span id="mhint"></span></figcaption>
    <div class="plot" id="plot"><svg id="svg" tabindex="0" role="img"></svg><div class="tip" id="tip" hidden></div></div>
  </figure>
  <details><summary>查看数据表</summary><div class="scroll"><table id="tbl"></table></div></details>
  <p class="note">学生 1 步采样，老师 4 步；基准是水平线（老师：虚线，iRDM s180：点线）。把鼠标放在图上，或者聚焦图表后用 ← → 键，可以看每个 step 所有已选模型的数值。种子多样性同时给出相对老师的倍数。</p>
</div>
<script type="application/json" id="data">__DATA__</script>
<script>
(function () {
  "use strict";
  var D = JSON.parse(document.getElementById("data").textContent);
  var NS = "http://www.w3.org/2000/svg";
  var SHAPES = ["circle", "square", "diamond", "triangle", "triangleDown", "pentagon", "cross"];
  var store = {
    get: function (k, d) { try { var v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
    set: function (k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) {} }
  };
  var runKeys = D.runs.map(function (r) { return r.key; });
  var baseKeys = D.baselines.map(function (b) { return b.key; });
  var allKeys = runKeys.concat(baseKeys);
  var metric = store.get("rdmEval.metric", "geneval");
  if (!D.metrics.some(function (m) { return m.key === metric; })) metric = D.metrics[0].key;
  var shown = new Set((store.get("rdmEval.shown", allKeys) || allKeys).filter(function (k) { return allKeys.indexOf(k) >= 0; }));
  var hoverIdx = -1, lastSteps = [];

  function el(tag, attrs, parent, text) {
    var e = document.createElement(tag);
    for (var k in (attrs || {})) e.setAttribute(k, attrs[k]);
    if (text !== undefined) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  function sv(tag, attrs, parent, text) {
    var e = document.createElementNS(NS, tag);
    for (var k in (attrs || {})) e.setAttribute(k, attrs[k]);
    if (text !== undefined) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  function shapePath(shape, x, y, r) {
    var p = [], i, a;
    switch (shape) {
      case "square": return "M" + (x - r * .85) + "," + (y - r * .85) + "h" + (1.7 * r) + "v" + (1.7 * r) + "h" + (-1.7 * r) + "z";
      case "diamond": return "M" + x + "," + (y - r * 1.15) + "L" + (x + r * 1.15) + "," + y + "L" + x + "," + (y + r * 1.15) + "L" + (x - r * 1.15) + "," + y + "z";
      case "triangle": return "M" + x + "," + (y - r * 1.15) + "L" + (x + r * 1.1) + "," + (y + r * .8) + "L" + (x - r * 1.1) + "," + (y + r * .8) + "z";
      case "triangleDown": return "M" + x + "," + (y + r * 1.15) + "L" + (x + r * 1.1) + "," + (y - r * .8) + "L" + (x - r * 1.1) + "," + (y - r * .8) + "z";
      case "pentagon":
        for (i = 0; i < 5; i++) { a = -Math.PI / 2 + i * 2 * Math.PI / 5; p.push((x + r * 1.1 * Math.cos(a)) + "," + (y + r * 1.1 * Math.sin(a))); }
        return "M" + p.join("L") + "z";
      case "cross": var t = r * .42, s = r * 1.05;
        return "M" + (x - t) + "," + (y - s) + "h" + (2 * t) + "v" + (s - t) + "h" + (s - t) + "v" + (2 * t) + "h" + (t - s) + "v" + (s - t) + "h" + (-2 * t) + "v" + (t - s) + "h" + (t - s) + "v" + (-2 * t) + "h" + (s - t) + "z";
      default: return "M" + (x - r) + "," + y + "a" + r + "," + r + " 0 1,0 " + (2 * r) + ",0a" + r + "," + r + " 0 1,0 " + (-2 * r) + ",0";
    }
  }
  function metaOf(key) { return D.metrics.filter(function (m) { return m.key === key; })[0]; }
  function fmt(v) { return v === null || v === undefined ? "–" : v.toFixed(metaOf(metric).digits); }
  function teacherVal() {
    var t = D.baselines.filter(function (b) { return b.key === "teacher_4step"; })[0];
    return t ? t.values[metric] : null;
  }
  function withRatio(v) {
    var t = teacherVal();
    if (metric.indexOf("div_") !== 0 || v === null || v === undefined || !t) return fmt(v);
    return fmt(v) + "  (" + (v / t).toFixed(2) + "×)";
  }
  function niceTicks(lo, hi, n) {
    var span = hi - lo || 1, step = Math.pow(10, Math.floor(Math.log10(span / n))), err = span / n / step;
    if (err >= 7.5) step *= 10; else if (err >= 3.5) step *= 5; else if (err >= 1.5) step *= 2;
    var out = [], v = Math.ceil(lo / step) * step;
    for (; v <= hi + step * 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  function baseClass(i) { return i === 0 ? "b1" : "b2"; }
  function baseDash(i) { return i === 0 ? "7 5" : "2 4"; }

  // ---------------- controls
  var mBox = document.getElementById("metrics");
  D.metrics.forEach(function (m) {
    var b = el("button", { type: "button", "aria-pressed": String(m.key === metric) }, mBox, m.label);
    b.addEventListener("click", function () {
      metric = m.key; store.set("rdmEval.metric", metric);
      Array.prototype.forEach.call(mBox.children, function (c) { c.setAttribute("aria-pressed", String(c === b)); });
      render();
    });
  });
  var sBox = document.getElementById("series");
  function key(parent, i, isBase) {
    var s = sv("svg", { width: 30, height: 14, "aria-hidden": "true" }, parent);
    if (isBase) { sv("line", { x1: 1, y1: 7, x2: 29, y2: 7, class: "base " + baseClass(i), "stroke-dasharray": baseDash(i) }, s); }
    else {
      sv("line", { x1: 1, y1: 7, x2: 29, y2: 7, class: "ln s" + (i % 7 + 1) }, s);
      sv("path", { d: shapePath(SHAPES[i % 7], 15, 7, 4), class: "mk s" + (i % 7 + 1) }, s);
    }
  }
  function addToggle(k, label, desc, i, isBase) {
    var lab = el("label", { title: desc }, sBox);
    var cb = el("input", { type: "checkbox" }, lab);
    cb.checked = shown.has(k);
    cb.addEventListener("change", function () {
      if (cb.checked) shown.add(k); else shown.delete(k);
      store.set("rdmEval.shown", Array.from(shown)); render();
    });
    key(lab, i, isBase);
    el("span", { class: "t" }, lab, label);
  }
  D.runs.forEach(function (r, i) { addToggle(r.key, r.label, r.desc, i, false); });
  D.baselines.forEach(function (b, i) { addToggle(b.key, b.label, b.desc, i, true); });
  function setShown(keys) {
    shown = new Set(keys); store.set("rdmEval.shown", keys);
    Array.prototype.forEach.call(sBox.querySelectorAll("input"), function (cb, i) { cb.checked = shown.has(allKeys[i]); });
    render();
  }
  document.getElementById("all").addEventListener("click", function () { setShown(allKeys.slice()); });
  document.getElementById("none").addEventListener("click", function () { setShown([]); });
  document.getElementById("onlyb").addEventListener("click", function () { setShown(baseKeys.slice()); });
  var themeBtn = document.getElementById("theme"), themes = ["auto", "light", "dark"], themeNames = { auto: "自动", light: "浅色", dark: "深色" };
  var theme = store.get("rdmEval.theme", "auto");
  function applyTheme() {
    if (theme === "auto") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", theme);
    themeBtn.textContent = "主题：" + themeNames[theme];
  }
  themeBtn.addEventListener("click", function () { theme = themes[(themes.indexOf(theme) + 1) % 3]; store.set("rdmEval.theme", theme); applyTheme(); });
  applyTheme();
  document.getElementById("sub").textContent = "生成于 " + D.created + " · 评测 GPU：" + D.gpu +
    " · " + D.runs.length + " 个训练 + " + D.baselines.length + " 个基准";

  // ---------------- chart
  var svg = document.getElementById("svg"), tip = document.getElementById("tip"), plot = document.getElementById("plot");
  var geom = null;

  function visibleRuns() {
    return D.runs.map(function (r, i) { return { r: r, i: i }; }).filter(function (o) {
      return shown.has(o.r.key) && o.r.points.some(function (p) { return p[metric] !== null && p[metric] !== undefined; });
    });
  }
  function visibleBases() {
    return D.baselines.map(function (b, i) { return { b: b, i: i }; }).filter(function (o) {
      return shown.has(o.b.key) && o.b.values[metric] !== null && o.b.values[metric] !== undefined;
    });
  }

  function render() {
    var m = metaOf(metric);
    document.getElementById("mtitle").textContent = m.label;
    document.getElementById("mhint").textContent = m.hint;
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    tip.hidden = true;
    var W = Math.max(plot.clientWidth || 960, 320), narrow = W < 640;
    var H = narrow ? 320 : 430, M = { l: 58, r: narrow ? 14 : 190, t: 14, b: 42 };
    svg.setAttribute("viewBox", "0 0 " + W + " " + H); svg.setAttribute("height", H);
    var runs = visibleRuns(), bases = visibleBases();
    var ys = [], xs = [];
    runs.forEach(function (o) { o.r.points.forEach(function (p) { if (p[metric] !== null && p[metric] !== undefined) { ys.push(p[metric]); xs.push(p.step); } }); });
    bases.forEach(function (o) { ys.push(o.b.values[metric]); });
    svg.setAttribute("aria-label", m.label + "，" + runs.length + " 个模型、" + bases.length + " 个基准");
    if (!ys.length) {
      sv("text", { x: W / 2, y: H / 2, "text-anchor": "middle", class: "empty" }, svg, "没有选中的模型，或者这个指标还没有数据");
      geom = null; renderTable(runs, bases, []); return;
    }
    var xmax = xs.length ? Math.max.apply(null, xs) : 2000, xmin = 0;
    var ylo = Math.min.apply(null, ys), yhi = Math.max.apply(null, ys), pad = (yhi - ylo) * 0.08 || Math.abs(yhi) * 0.02 || 0.01;
    ylo -= pad; yhi += pad;
    var X = function (v) { return M.l + (v - xmin) / (xmax - xmin || 1) * (W - M.l - M.r); };
    var Y = function (v) { return H - M.b - (v - ylo) / (yhi - ylo) * (H - M.t - M.b); };
    var yt = niceTicks(ylo, yhi, 5), xt = niceTicks(xmin, xmax, narrow ? 4 : 8);
    yt.forEach(function (v) {
      sv("line", { x1: M.l, x2: W - M.r, y1: Y(v), y2: Y(v), class: "gridln" }, svg);
      sv("text", { x: M.l - 8, y: Y(v) + 4, "text-anchor": "end", class: "tick" }, svg, v.toFixed(m.digits > 3 ? 3 : 2));
    });
    sv("line", { x1: M.l, x2: W - M.r, y1: H - M.b, y2: H - M.b, class: "axisln" }, svg);
    xt.forEach(function (v) { sv("text", { x: X(v), y: H - M.b + 16, "text-anchor": "middle", class: "tick" }, svg, String(v)); });
    sv("text", { x: (M.l + W - M.r) / 2, y: H - 6, "text-anchor": "middle", class: "axl" }, svg, "训练步数（step）");
    var labels = [];
    bases.forEach(function (o) {
      var y = Y(o.b.values[metric]);
      sv("line", { x1: M.l, x2: W - M.r, y1: y, y2: y, class: "base " + baseClass(o.i), "stroke-dasharray": baseDash(o.i) }, svg);
      labels.push({ y: y, text: o.b.label, val: withRatio(o.b.values[metric]), cls: "base " + baseClass(o.i), dash: baseDash(o.i), base: true });
    });
    runs.forEach(function (o) {
      var pts = o.r.points.filter(function (p) { return p[metric] !== null && p[metric] !== undefined; });
      var cls = "s" + (o.i % 7 + 1);
      sv("path", { d: pts.map(function (p, j) { return (j ? "L" : "M") + X(p.step) + "," + Y(p[metric]); }).join(""), class: "ln " + cls }, svg);
      pts.forEach(function (p) { sv("path", { d: shapePath(SHAPES[o.i % 7], X(p.step), Y(p[metric]), 4.3), class: "mk " + cls }, svg); });
      var last = pts[pts.length - 1];
      labels.push({ y: Y(last[metric]), text: o.r.label, val: withRatio(last[metric]), cls: cls, shape: SHAPES[o.i % 7] });
    });
    if (!narrow) {                       // direct labels at the right edge, pushed apart vertically
      labels.sort(function (a, b) { return a.y - b.y; });
      for (var k = 1; k < labels.length; k++) if (labels[k].y - labels[k - 1].y < 26) labels[k].y = labels[k - 1].y + 26;
      var over = labels.length ? labels[labels.length - 1].y - (H - M.b) : 0;
      if (over > 0) labels.forEach(function (l) { l.y -= over; });
      labels.forEach(function (l) {
        var x0 = W - M.r + 10;
        if (l.base) sv("line", { x1: x0, x2: x0 + 14, y1: l.y - 3, y2: l.y - 3, class: l.cls, "stroke-dasharray": l.dash }, svg);
        else sv("path", { d: shapePath(l.shape, x0 + 6, l.y - 3, 4), class: "mk " + l.cls }, svg);
        sv("text", { x: x0 + 20, y: l.y, class: "dl" }, svg, l.text);
        sv("text", { x: x0 + 20, y: l.y + 12, class: "dlv" }, svg, l.val);
      });
    }
    var steps = Array.from(new Set(xs)).sort(function (a, b) { return a - b; });
    geom = { X: X, Y: Y, M: M, W: W, H: H, steps: steps, runs: runs, bases: bases };
    lastSteps = steps;
    if (hoverIdx >= steps.length) hoverIdx = steps.length - 1;
    renderTable(runs, bases, steps);
  }

  function showAt(idx) {
    if (!geom || !geom.steps.length) return;
    hoverIdx = Math.max(0, Math.min(idx, geom.steps.length - 1));
    var step = geom.steps[hoverIdx], x = geom.X(step);
    var old = svg.querySelector(".cross"); if (old) old.remove();
    sv("line", { x1: x, x2: x, y1: geom.M.t, y2: geom.H - geom.M.b, class: "cross" }, svg);
    while (tip.firstChild) tip.removeChild(tip.firstChild);
    el("div", { class: "h" }, tip, "step " + step);
    geom.runs.forEach(function (o) {
      var p = o.r.points.filter(function (q) { return q.step === step; })[0];
      var row = el("div", { class: "r" }, tip);
      var k = sv("svg", { width: 16, height: 10, "aria-hidden": "true" }, row);
      sv("line", { x1: 1, y1: 5, x2: 15, y2: 5, class: "ln s" + (o.i % 7 + 1) }, k);
      el("b", {}, row, p ? withRatio(p[metric]) : "–");
      el("span", {}, row, o.r.label);
    });
    geom.bases.forEach(function (o) {
      var row = el("div", { class: "r" }, tip);
      var k = sv("svg", { width: 16, height: 10, "aria-hidden": "true" }, row);
      sv("line", { x1: 1, y1: 5, x2: 15, y2: 5, class: "base " + baseClass(o.i), "stroke-dasharray": baseDash(o.i) }, k);
      el("b", {}, row, withRatio(o.b.values[metric]));
      el("span", {}, row, o.b.label);
    });
    tip.hidden = false;
    var pw = plot.clientWidth || geom.W, scale = pw / geom.W, px = x * scale, tw = tip.offsetWidth || 220;
    tip.style.left = (px + 14 + tw > pw ? Math.max(0, px - 14 - tw) : px + 14) + "px";
  }
  function hide() { var old = svg.querySelector(".cross"); if (old) old.remove(); tip.hidden = true; }
  svg.addEventListener("pointermove", function (e) {
    if (!geom || !geom.steps.length) return;
    var r = svg.getBoundingClientRect(), x = (e.clientX - r.left) / (r.width || geom.W) * geom.W, best = 0;
    geom.steps.forEach(function (s, i) { if (Math.abs(geom.X(s) - x) < Math.abs(geom.X(geom.steps[best]) - x)) best = i; });
    showAt(best);
  });
  svg.addEventListener("pointerleave", hide);
  svg.addEventListener("keydown", function (e) {
    if (e.key === "ArrowRight") { showAt(hoverIdx < 0 ? 0 : hoverIdx + 1); e.preventDefault(); }
    else if (e.key === "ArrowLeft") { showAt(hoverIdx < 0 ? lastSteps.length - 1 : hoverIdx - 1); e.preventDefault(); }
    else if (e.key === "Escape") hide();
  });
  svg.addEventListener("blur", hide);

  function renderTable(runs, bases, steps) {
    var t = document.getElementById("tbl");
    while (t.firstChild) t.removeChild(t.firstChild);
    var head = el("tr", {}, el("thead", {}, t));
    el("th", {}, head, "step");
    runs.forEach(function (o) { el("th", {}, head, o.r.label); });
    bases.forEach(function (o) { el("th", {}, head, o.b.label); });
    var body = el("tbody", {}, t);
    steps.forEach(function (s) {
      var tr = el("tr", {}, body);
      el("td", {}, tr, String(s));
      runs.forEach(function (o) { var p = o.r.points.filter(function (q) { return q.step === s; })[0]; el("td", {}, tr, p ? fmt(p[metric]) : "–"); });
      bases.forEach(function (o) { el("td", {}, tr, fmt(o.b.values[metric])); });
    });
  }

  var raf = 0;
  window.addEventListener("resize", function () { cancelAnimationFrame(raf); raf = requestAnimationFrame(render); });
  render();
})();
</script>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    if not os.path.isfile(os.path.join(out, "models.json")):
        raise SystemExit(f"{out}/models.json not found -- run scripts/eval_hf_models.sh first")
    meta, runs, baselines = collect(out)
    gpu = gpu_name()
    for key, ylabel, title, name in [
            ("geneval", "GenEval (overall)", "GenEval vs training step", "fig_geneval.png"),
            ("pickscore", "PickScore", "PickScore vs training step", "fig_pickscore.png"),
            ("div_dreamsim", "seed diversity (DreamSim, 1 - cos)", "Seed diversity (DreamSim) vs training step",
             "fig_div_dreamsim.png")]:
        if not figure(os.path.join(out, name), runs, baselines, key, ylabel, title):
            try:
                os.remove(os.path.join(out, name))
            except FileNotFoundError:
                pass
    write_markdown(out, meta, runs, baselines, gpu)
    write_csv(out, runs, baselines)
    write_html(out, runs, baselines, gpu)
    n = sum(len(r["points"]) for r in runs) + sum(1 for b in baselines if b["values"])
    print(f"[report] {n} evaluation(s) -> {out}/report.md, {out}/curves.html, {out}/results.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
