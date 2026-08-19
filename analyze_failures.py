"""失败案例分析 — 找出模型集成与人工标注不一致的样本，定位系统性盲区。

输出：
  analysis/failure_report.csv   全部不一致样本（投票、置信度、错误方向）
  analysis/representative.csv   每类高置信正确样本（作为分级标准典型示例）
  analysis/标准示例图.png       每类 4 张典型示例拼图
"""
from __future__ import annotations

from pathlib import Path
from collections import Counter
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from PIL import Image, ImageDraw

from model_utils import (load_ensemble, INFERENCE_TRANSFORM,
                         NUM_CLASSES, DEVICE, CLASS_NAMES)

BASE = Path(__file__).parent
ANNOTATED = BASE / "data" / "cache_1024"
OUT_DIR = BASE / "analysis"

GRADE = {1: "一级", 2: "二级", 3: "三级"}

transform = INFERENCE_TRANSFORM

# ─── 加载全部模型（使用公共模块） ───
ms = load_ensemble()
print(f"已加载 {len(ms)} 个模型")

# ─── 预测全部 600 张 ───
ds = datasets.ImageFolder(root=str(ANNOTATED), transform=transform)
loader = DataLoader(ds, batch_size=64, shuffle=False)

all_probs = []  # (N, nmodels, nclasses)
with torch.no_grad():
    for x, _ in loader:
        x = x.to(DEVICE)
        batch = []
        for m in ms:
            batch.append(torch.softmax(m(x), dim=1).cpu().numpy())
        all_probs.append(np.stack(batch, axis=1))
all_probs = np.concatenate(all_probs, axis=0)

avg_probs = all_probs.mean(axis=1)          # (N, nclasses) 软投票
votes = all_probs.argmax(axis=2)            # (N, nmodels) 每模型硬票
labels = np.array(ds.targets)               # (N,)

preds = avg_probs.argmax(axis=1)
conf = avg_probs.max(axis=1)
names = [Path(p).name for p, _ in ds.imgs]

# ─── 不一致样本 ───
disagree_idx = np.where(preds != labels)[0]
print(f"\n人机不一致: {len(disagree_idx)}/{len(labels)} 张")

rows = []
for i in disagree_idx:
    v = Counter(votes[i])
    vote_str = " ".join(f"{GRADE[k+1]}:{v[k]}" for k in range(NUM_CLASSES) if v[k])
    rows.append({
        "文件名": names[i],
        "人标": GRADE[labels[i] + 1],
        "模型判": GRADE[preds[i] + 1],
        "置信度": f"{conf[i]:.3f}",
        "投票": vote_str,
        "类别": "高置信判错" if conf[i] >= 0.60 else "边界分歧",
    })

# 按置信度降序（高置信判错排前面，最可疑）
rows.sort(key=lambda r: float(r["置信度"]), reverse=True)

# ─── 每类典型示例（高置信且正确） ───
correct = np.where(preds == labels)[0]
rep_rows = []
for c in range(NUM_CLASSES):
    idx_in_class = correct[labels[correct] == c]
    order = np.argsort(-conf[idx_in_class])[:4]
    for o in order:
        i = idx_in_class[o]
        rep_rows.append({"类别": GRADE[c + 1], "文件名": names[i], "置信度": f"{conf[i]:.3f}"})

# ─── 写 CSV ───
OUT_DIR.mkdir(exist_ok=True)
csv_path = OUT_DIR / "failure_report.csv"
with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["文件名", "人标", "模型判", "置信度", "投票", "类别"])
    w.writeheader()
    w.writerows(rows)

rep_path = OUT_DIR / "representative.csv"
with open(rep_path, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["类别", "文件名", "置信度"])
    w.writeheader()
    w.writerows(rep_rows)

# ─── 生成标准示例图（每类 4 张拼图） ───
CELL, MARGIN, LABEL_H = 360, 8, 36
cols, rows_per_class = 4, 3
W = cols * CELL + (cols + 1) * MARGIN
H = rows_per_class * (CELL + LABEL_H) + (rows_per_class + 1) * MARGIN + LABEL_H
sheet = Image.new("RGB", (W, H), "#1f1f1f")
draw = ImageDraw.Draw(sheet)

rep_by_class = {}
for r in rep_rows:
    rep_by_class.setdefault(r["类别"], []).append(r["文件名"])

for ci, cls in enumerate(["一级", "二级", "三级"]):
    y0 = ci * (CELL + LABEL_H + MARGIN) + MARGIN
    draw.text((MARGIN + 6, y0 + 6), f"{cls}:", fill="#ffffff")
    for k, fname in enumerate(rep_by_class.get(cls, [])[:cols]):
        img = Image.open(ANNOTATED / str(ci + 1) / fname).convert("RGB")
        img.thumbnail((CELL, CELL))
        x = MARGIN + k * (CELL + MARGIN)
        sheet.paste(img, (x, y0 + LABEL_H), mask=None)

img_path = OUT_DIR / "标准示例图.png"
sheet.save(img_path)
print(f"\n已保存: {csv_path.name} ({len(rows)} 张) / {rep_path.name} / {img_path.name}")

# ─── 控制台汇总 ───
print(f"\n{'='*46}")
print(f"  失败分析汇总")
print(f"{'='*46}")
print(f"\n  不一致总数: {len(rows)}")
from collections import Counter as C2
dirs = C2(f"{r['人标']}→{r['模型判']}" for r in rows)
for k, v in dirs.most_common():
    print(f"    {k}: {v} 张")
print(f"\n  高置信判错（置信度>=0.6，最可疑）: "
      f"{sum(1 for r in rows if r['类别']=='高置信判错')} 张")
print(f"  边界分歧（置信度<0.6，属于标准模糊带）: "
      f"{sum(1 for r in rows if r['类别']=='边界分歧')} 张")

print("\n  不一致样本明细:")
for r in rows:
    print(f"    {r['文件名']}  人标{r['人标']} 模型{r['模型判']}  "
          f"conf={r['置信度']}  [{r['类别']}]  {r['投票']}")
