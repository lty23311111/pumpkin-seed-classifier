"""南瓜子3分类边界复查 — 找出人和模型标注不一致的图片，逐个重标。

用模型投票找出与人工标注冲突的样本，逐张重新判定。标注不准的地方会集中暴露在这里。
"""
from pathlib import Path
from collections import Counter
import shutil
import tkinter as tk
from tkinter import messagebox

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from PIL import Image, ImageTk

from model_utils import load_ensemble, INFERENCE_TRANSFORM, NUM_CLASSES, DEVICE
from ui_theme import GRADE_COLORS_1, FONT

BASE = Path(__file__).parent
RAW = BASE / "data" / "raw"
ANNOTATED = BASE / "data" / "annotated_3class"

CLASS_NAMES = {1: "一级·完好", 2: "二级·轻微瑕疵", 3: "三级·明显瑕疵"}
GRADE_COLORS = GRADE_COLORS_1

transform = INFERENCE_TRANSFORM

# 加载模型：使用公共模块（与 gui.py 一致）
_models = load_ensemble()
n_models = len(_models)

# 用投票方式预测
print("预测中...")
dataset = datasets.ImageFolder(root=str(ANNOTATED), transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False)
all_votes = []
with torch.no_grad():
    for inputs, _ in loader:
        inputs = inputs.to(DEVICE)
        batch_votes = []
        for m in _models:
            out = m(inputs)
            batch_votes.append(out.argmax(1).cpu().tolist())
        batch_votes = list(zip(*batch_votes))
        all_votes.extend(batch_votes)

disagreements = []
for (path_str, human_label), votes in zip(dataset.imgs, all_votes):
    name = Path(path_str).name
    human_label_1 = human_label + 1  # 转为 1-indexed
    counter = Counter(votes)
    pred_label_0 = counter.most_common(1)[0][0]
    pred_label_1 = pred_label_0 + 1
    n_agree = counter[pred_label_0]

    if human_label_1 != pred_label_1:
        raw_path = RAW / name
        if raw_path.exists():
            disagreements.append({
                "name": name,
                "human": human_label_1,
                "predicted": pred_label_1,
                "agreement": f"{n_agree}/{n_models}",
                "votes": {k + 1: v for k, v in counter.items()},
            })

disagreements.sort(key=lambda d: abs(d["human"] - d["predicted"]), reverse=True)
print(f"不一致: {len(disagreements)} 张")

if not disagreements:
    print("全部一致！")
    exit()

# ─── 审核界面 ───
MAX_W, MAX_H = 1200, 800


class Reviewer:
    def __init__(self, items):
        self.items = items
        self.total = len(items)
        self.idx = 0
        self.changes = {}

        self.root = tk.Tk()
        self.root.title(f"3分类复查 — 共 {self.total} 张不一致")
        self.root.geometry(f"{MAX_W}x{MAX_H + 100}")
        self.root.configure(bg="#2b2b2b")

        self.info_var = tk.StringVar()
        self.human_var = tk.StringVar()
        self.pred_var = tk.StringVar()
        self.vote_var = tk.StringVar()

        info_frame = tk.Frame(self.root, bg="#2b2b2b", pady=5)
        info_frame.pack(side=tk.TOP, fill=tk.X)
        tk.Label(info_frame, textvariable=self.info_var, fg="#aaa", bg="#2b2b2b",
                 font=("Consolas", 12)).pack(side=tk.LEFT, padx=15)
        tk.Label(info_frame, textvariable=self.human_var, fg="#ffcc00", bg="#2b2b2b",
                 font=("Consolas", 12, "bold")).pack(side=tk.LEFT, padx=15)
        tk.Label(info_frame, textvariable=self.pred_var, fg="#ff9800", bg="#2b2b2b",
                 font=("Consolas", 12, "bold")).pack(side=tk.LEFT, padx=15)
        tk.Label(info_frame, textvariable=self.vote_var, fg="#aaa", bg="#2b2b2b",
                 font=("Consolas", 10)).pack(side=tk.LEFT, padx=15)

        self.image_label = tk.Label(self.root, bg="#1a1a1a")
        self.image_label.pack(fill=tk.BOTH, expand=True)

        hint_frame = tk.Frame(self.root, bg="#2b2b2b", pady=8)
        hint_frame.pack(side=tk.BOTTOM, fill=tk.X)
        for g, name, color in zip(range(1, NUM_CLASSES + 1), CLASS_NAMES.values(), GRADE_COLORS.values()):
            tk.Label(hint_frame, text=f"  [{g}] {name}  ", fg=color, bg="#2b2b2b",
                     font=("Consolas", 11, "bold")).pack(side=tk.LEFT)
        tk.Label(hint_frame, text="  [0] 保留原标  ", fg="#aaa", bg="#2b2b2b",
                 font=("Consolas", 11)).pack(side=tk.LEFT)
        tk.Label(hint_frame, text="  Backspace 回退  ", fg="#888", bg="#2b2b2b",
                 font=("Consolas", 11)).pack(side=tk.LEFT)
        tk.Label(hint_frame, text="  ESC 退出  ", fg="#888", bg="#2b2b2b",
                 font=("Consolas", 11)).pack(side=tk.LEFT)

        for g in (1, 2, 3):
            self.root.bind(str(g), lambda e, grade=g: self.reclassify(grade))
        self.root.bind("0", lambda e: self.keep())
        self.root.bind("<BackSpace>", lambda e: self.go_back())
        self.root.bind("<Escape>", lambda e: self.quit())
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

        self.show_current()

    def show_current(self):
        if self.idx >= self.total:
            messagebox.showinfo("完成", f"复查完成！纠正 {len(self.changes)} 张。\n重新运行 train_3class.py 训练。")
            self.root.destroy()
            return

        item = self.items[self.idx]
        self.info_var.set(f"[{self.idx + 1}/{self.total}]  {item['name']}")
        self.human_var.set(f"👤 人标: {CLASS_NAMES[item['human']]} ({item['human']})")
        self.pred_var.set(f"🤖 模型: {CLASS_NAMES[item['predicted']]} ({item['predicted']})")

        votes = item.get("votes", {})
        vote_parts = [f"{CLASS_NAMES[k]}:{v}票" for k, v in sorted(votes.items())]
        self.vote_var.set(f"投票: {'  '.join(vote_parts)}  ({item.get('agreement', '?')})")

        img = Image.open(str(RAW / item["name"]))
        w, h = img.size
        scale = min(MAX_W / w, MAX_H / h, 1.0)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self.photo)

    def reclassify(self, grade):
        item = self.items[self.idx]
        name = item["name"]
        old = item["human"]
        if grade != old:
            if not messagebox.askyesno("确认重标", f"将 {name} 改为 {CLASS_NAMES[grade]}？"):
                return
            destination = ANNOTATED / str(grade) / name
            # 先复制成功，再删除旧标签，避免复制失败时丢失样本。
            shutil.copy2(RAW / name, destination)
            for g in (1, 2, 3):
                existing = ANNOTATED / str(g) / name
                if existing.exists() and existing != destination:
                    existing.unlink()
            self.changes[name] = grade
            print(f"  {name}: {CLASS_NAMES[old]} → {CLASS_NAMES[grade]}")
        else:
            print(f"  {name}: 维持 {CLASS_NAMES[grade]}")
        self.idx += 1
        self.show_current()

    def keep(self):
        item = self.items[self.idx]
        print(f"  {item['name']}: 保留 {CLASS_NAMES[item['human']]}")
        self.idx += 1
        self.show_current()

    def go_back(self):
        if self.idx > 0:
            self.idx -= 1
        self.show_current()

    def quit(self):
        print(f"\n纠正 {len(self.changes)} 张，退出。")
        self.root.destroy()


Reviewer(disagreements).root.mainloop()

# 统计
counts = Counter()
for g in (1, 2, 3):
    counts[g] = len(list((ANNOTATED / str(g)).glob("*.bmp")))
print(f"\n========== 复查后分布 ==========")
for g in (1, 2, 3):
    print(f"  {CLASS_NAMES[g]}: {counts[g]} 张")
print(f"  总计: {sum(counts.values())} 张")
