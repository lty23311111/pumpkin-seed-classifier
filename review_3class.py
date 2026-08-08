"""南瓜子3分类边界复查 — 找出人和模型标注不一致的图片，逐个重标。

用模型投票找出与人工标注冲突的样本，逐张重新判定。标注不准的地方会集中暴露在这里。
"""
from pathlib import Path
from collections import Counter
import shutil
import tkinter as tk
from tkinter import messagebox

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from PIL import Image, ImageTk

BASE = Path(__file__).parent
RAW = BASE / "data" / "raw"
ANNOTATED = BASE / "data" / "annotated_3class"
MODEL_DIR = BASE / "models"

NUM_CLASSES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = {1: "一级·完好", 2: "二级·轻微瑕疵", 3: "三级·明显瑕疵"}
GRADE_COLORS = {1: "#4caf50", 2: "#ff9800", 3: "#ef5350"}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 加载模型：新高必入 + fold 补充投票（与 gui.py 一致）
_models: list[nn.Module] = []
model_files: list[Path] = []

best_path = MODEL_DIR / "best_model_3class.pth"
if best_path.exists():
    model_files.append(best_path)

for fp in sorted(MODEL_DIR.glob("fold_*.pth")):
    if fp not in model_files:
        model_files.append(fp)

for p in [MODEL_DIR / "best_model_3class_2.pth",
          MODEL_DIR / "best_model_3class_3.pth"]:
    if p.exists() and p not in model_files:
        model_files.append(p)

for mp in model_files:
    m = models.resnet50()
    m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(2048, NUM_CLASSES))
    m.load_state_dict(torch.load(mp, map_location=DEVICE, weights_only=True))
    m = m.to(DEVICE)
    m.eval()
    _models.append(m)

if not _models:
    raise FileNotFoundError("未找到任何模型文件，请先运行 train_3class.py")

n_models = len(_models)
print(f"已加载 {n_models} 个模型" + (" (投票)" if n_models > 1 else " (单模型)"))

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
