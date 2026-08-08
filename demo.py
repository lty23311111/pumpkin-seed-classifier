"""现场演示 — 拍图，模型预测 + 人工标注对比，实时计算准确率。"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from PIL import Image, ImageTk
from torchvision import models, transforms
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

BASE = Path(__file__).parent
MODEL_DIR = BASE / "models"
NUM_CLASSES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["一级 · 完好", "二级 · 轻微瑕疵", "三级 · 明显瑕疵"]
GRADE_COLORS = {0: "#4caf50", 1: "#ff9800", 2: "#ef5350"}
GRADE_BG     = {0: "#1e3520", 1: "#3d301e", 2: "#3d1e1e"}
C_BG, C_SFC, C_TXT, C_TXT2 = "#1f1f1f", "#2c2c2c", "#f0f0f0", "#7a7a7a"
FONT = "Microsoft YaHei UI"
MONO = "Cascadia Code"

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

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict(img_path: Path):
    img = Image.open(img_path).convert("RGB")
    t = _transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        all_preds = []
        all_probs = []
        for m in _models:
            out = m(t)
            all_preds.append(out.argmax(1).item())
            all_probs.append(torch.softmax(out, dim=1)[0].cpu().tolist())
    avg_probs = [(sum(p[i] for p in all_probs) / len(all_probs)) for i in range(NUM_CLASSES)]
    pred = max(range(NUM_CLASSES), key=avg_probs.__getitem__)
    return pred, avg_probs


class DemoApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("南瓜子外观质量分类 · 现场演示")
        self.root.geometry("1200x780")
        self.root.configure(bg=C_BG)
        self.root.minsize(1000, 640)

        self._images: list[Path] = []
        self._idx = 0
        self._results: list[dict] = []
        self._photo = None

        self._build_ui()

    def _build_ui(self):
        # ── 标题栏 ──
        bar = tk.Frame(self.root, bg=C_SFC, height=52)
        bar.pack(side=tk.TOP, fill=tk.X)
        bar.pack_propagate(False)
        tk.Label(bar, text="🎃  南瓜子外观质量分类 · 现场验收演示", fg=C_TXT, bg=C_SFC,
                 font=(FONT, 15, "bold")).pack(side=tk.LEFT, padx=20, pady=12)
        n_models = len(_models)
        model_badge = f"投票 ×{n_models}" if n_models > 1 else "单模型"
        tk.Label(bar, text=f"ResNet-50  ·  3 分类  ·  {model_badge}",
                 fg="#ff8c00", bg=C_SFC, font=(FONT, 9)).pack(side=tk.LEFT, padx=8, pady=14)
        self._score_label = tk.Label(bar, text="", fg="#ff8c00", bg=C_SFC, font=(FONT, 12, "bold"))
        self._score_label.pack(side=tk.RIGHT, padx=20, pady=12)

        # ── 主体 ──
        body = tk.Frame(self.root, bg=C_BG)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=16, pady=(14, 6))

        # 左：图片
        left = tk.Frame(body, bg=C_SFC, width=600)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left.pack_propagate(False)

        self._img_area = tk.Frame(left, bg="#252525", cursor="hand2")
        self._img_area.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self._img_area.bind("<Button-1>", lambda e: self._choose_folder())

        self._placeholder = tk.Frame(self._img_area, bg="#252525")
        self._placeholder.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        ph_icon = tk.Label(self._placeholder, text="📷", font=("Segoe UI Emoji", 56), bg="#252525")
        ph_icon.pack()
        ph_text = tk.Label(self._placeholder, text="点击此处选择图片文件夹", fg="#666",
                           bg="#252525", font=(FONT, 16, "bold"))
        ph_text.pack(pady=(10, 4))
        ph_hint = tk.Label(self._placeholder, text="将照片放入文件夹 → 点击选择开始",
                           fg="#444", bg="#252525", font=(FONT, 11))
        ph_hint.pack()
        for w in (self._placeholder, ph_icon, ph_text, ph_hint):
            w.bind("<Button-1>", lambda e: self._choose_folder())

        self._img_label = tk.Label(self._img_area, bg="#252525")
        self._img_label.bind("<Button-1>", lambda e: self._choose_folder())

        # 右：控制 + 结果
        right = tk.Frame(body, bg=C_BG, width=560)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(14, 0))
        right.pack_propagate(False)

        # 进度区
        nav_card = tk.Frame(right, bg=C_SFC)
        nav_card.pack(fill=tk.X, pady=(0, 8))
        ni = tk.Frame(nav_card, bg=C_SFC); ni.pack(padx=20, pady=14, fill=tk.X)

        self._nav_label = tk.Label(ni, text="等待选择文件夹...", fg=C_TXT,
                                   bg=C_SFC, font=(FONT, 14, "bold"))
        self._nav_label.pack(side=tk.LEFT)

        self._progress = ttk.Progressbar(ni, length=160, mode="determinate")
        self._progress.pack(side=tk.RIGHT)

        # 模型预测区
        pred_card = tk.Frame(right, bg=C_SFC)
        pred_card.pack(fill=tk.X, pady=(0, 8))
        pi = tk.Frame(pred_card, bg=C_SFC); pi.pack(padx=20, pady=14, fill=tk.X)
        tk.Label(pi, text="🤖  模型预测", fg=C_TXT2, bg=C_SFC, font=(FONT, 10, "bold")).pack(anchor=tk.W)

        self._pred_frame = tk.Frame(pi, bg=GRADE_BG[0])
        self._pred_frame.pack(fill=tk.X, pady=(8, 2))
        self._pred_label = tk.Label(self._pred_frame, text="", fg=GRADE_COLORS[0],
                                    bg=GRADE_BG[0], font=(FONT, 22, "bold"))
        self._pred_label.pack(side=tk.LEFT, padx=20, pady=12)
        self._pred_prob = tk.Label(pi, text="", fg=C_TXT2, bg=C_SFC, font=(FONT, 12))
        self._pred_prob.pack(anchor=tk.W, pady=(4, 0))

        # 人工标注区
        human_card = tk.Frame(right, bg=C_SFC)
        human_card.pack(fill=tk.X, pady=(0, 8))
        hi = tk.Frame(human_card, bg=C_SFC); hi.pack(padx=20, pady=14, fill=tk.X)
        tk.Label(hi, text="👤  人工标注（请按键）", fg=C_TXT2, bg=C_SFC,
                 font=(FONT, 10, "bold")).pack(anchor=tk.W)

        self._human_label = tk.Label(hi, text="尚未标注", fg=C_TXT2, bg=C_SFC,
                                     font=(FONT, 16))
        self._human_label.pack(anchor=tk.W, pady=(8, 2))

        # 三个等级按钮
        btn_row = tk.Frame(right, bg=C_BG)
        btn_row.pack(fill=tk.X, pady=(0, 4))
        colors_hex = ["#4caf50", "#ff9800", "#ef5350"]
        names_short = ["一级·完好", "二级·轻微", "三级·明显"]
        keys = ["1", "2", "3"]
        self._grade_btns = []
        for g in range(NUM_CLASSES):
            b = tk.Button(btn_row, text=f"  [{keys[g]}]  {names_short[g]}",
                          command=lambda g0=g: self._mark_human(g0),
                          bg=colors_hex[g], fg="#fff",
                          font=(FONT, 10, "bold"), relief=tk.FLAT,
                          cursor="hand2", padx=10, pady=6)
            b.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
            b.bind("<Enter>", lambda e, btn=b, c=g: btn.configure(bg=self._lighter(colors_hex[c])))
            b.bind("<Leave>", lambda e, btn=b, c=g: btn.configure(bg=colors_hex[c]))
            self._grade_btns.append(b)
            self.root.bind(keys[g], lambda e, g0=g: self._mark_human(g0))

        # 导航按钮
        nav_row = tk.Frame(right, bg=C_BG)
        nav_row.pack(fill=tk.X, pady=(4, 0))
        self._btn_prev = tk.Button(nav_row, text="◀  上一张", command=self._prev,
                                   bg="#444", fg="#ddd", font=(FONT, 10), relief=tk.FLAT,
                                   cursor="hand2", padx=16, pady=6, state=tk.DISABLED)
        self._btn_prev.pack(side=tk.LEFT)
        self._btn_next = tk.Button(nav_row, text="下一张  ▶", command=self._next,
                                   bg=C_SFC, fg=C_TXT, font=(FONT, 10, "bold"), relief=tk.FLAT,
                                   cursor="hand2", padx=16, pady=6, state=tk.DISABLED)
        self._btn_next.pack(side=tk.RIGHT)

        # 换文件夹
        tk.Button(nav_row, text="📁  选择文件夹", command=self._choose_folder,
                  bg=C_SFC, fg=C_TXT2, font=(FONT, 9), relief=tk.FLAT,
                  cursor="hand2", padx=10, pady=6).pack(side=tk.RIGHT, padx=(0, 8))

        # 底部状态
        s_bar = tk.Frame(self.root, bg=C_SFC, height=30)
        s_bar.pack(side=tk.BOTTOM, fill=tk.X)
        s_bar.pack_propagate(False)
        self._status = tk.Label(s_bar, text="就绪 · 请选择图片文件夹", fg=C_TXT2,
                                bg=C_SFC, anchor=tk.W, font=(FONT, 9))
        self._status.pack(side=tk.LEFT, padx=18, pady=5)

    @staticmethod
    def _lighter(hex_color):
        r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
        r, g, b = min(255, r+40), min(255, g+40), min(255, b+40)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _choose_folder(self):
        folder = filedialog.askdirectory(title="选择图片文件夹")
        if not folder:
            return
        folder = Path(folder)
        images = sorted(list(folder.glob("*.bmp")) + list(folder.glob("*.jpg")) +
                        list(folder.glob("*.jpeg")) + list(folder.glob("*.png")))
        if not images:
            messagebox.showinfo("提示", "文件夹中没有图片文件")
            return

        self._images = images
        self._idx = 0
        self._results = []

        # 预跑模型预测
        self._status.configure(text=f"模型预测中... 0/{len(images)}")
        self._nav_label.configure(text=f"共 {len(images)} 张  ·  预测中...")

        def _run():
            for i, img in enumerate(images):
                pred, probs = predict(img)
                self._results.append({"name": img.name, "model": pred,
                                      "probs": probs, "human": None})
                self.root.after(0, lambda c=i+1: [
                    self._progress.configure(value=100 * c / len(images)),
                    self._status.configure(text=f"模型预测: {c}/{len(images)}"),
                ])
            self.root.after(0, self._show_current)

        threading.Thread(target=_run, daemon=True).start()

    def _show_current(self):
        total = len(self._images)
        if total == 0:
            return
        self._idx = max(0, min(self._idx, total - 1))
        r = self._results[self._idx]

        # 图片
        img = Image.open(self._images[self._idx]).convert("RGB")
        max_w, max_h = 580, 480
        w, h = img.size
        s = min(max_w / w, max_h / h, 1.0)
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)
        self._img_label.configure(image=self._photo, text="")
        self._img_label.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self._placeholder.place_forget()

        # 模型预测
        pred, probs = r["model"], r["probs"]
        self._pred_frame.configure(bg=GRADE_BG[pred])
        self._pred_label.configure(text=CLASS_NAMES[pred], fg=GRADE_COLORS[pred],
                                   bg=GRADE_BG[pred])
        self._pred_prob.configure(text=f"置信度  {probs[pred]:.1%}")

        # 人工标注
        human = r["human"]
        if human is not None:
            self._human_label.configure(text=f"✅  {CLASS_NAMES[human]}",
                                        fg=GRADE_COLORS[human])
        else:
            self._human_label.configure(text="尚未标注", fg=C_TXT2)

        # 导航
        self._nav_label.configure(text=f"第 {self._idx + 1} / {total} 张")
        self._btn_prev.configure(state=tk.NORMAL if self._idx > 0 else tk.DISABLED)
        self._btn_next.configure(state=tk.NORMAL if self._idx < total - 1 else tk.DISABLED)

        # 实时准确率
        labeled = [r for r in self._results if r["human"] is not None]
        if labeled:
            correct = sum(1 for r in labeled if r["human"] == r["model"])
            acc = correct / len(labeled)
            self._score_label.configure(
                text=f"📊  一致率 {correct}/{len(labeled)} = {acc:.1%}")
        else:
            self._score_label.configure(text="")

        self._status.configure(text=f"{self._images[self._idx].name}  ·  [{self._idx + 1}/{total}]")

    def _mark_human(self, grade: int):
        if not self._results:
            return
        self._results[self._idx]["human"] = grade
        self._show_current()
        # 如果全部标完，自动弹窗
        if all(r["human"] is not None for r in self._results):
            self._show_summary()

    def _prev(self):
        self._idx -= 1
        self._show_current()

    def _next(self):
        self._idx += 1
        self._show_current()

    def _show_summary(self):
        total = len(self._results)
        correct = sum(1 for r in self._results if r["human"] == r["model"])
        acc = correct / total
        human_counts = Counter(r["human"] for r in self._results)
        model_counts = Counter(r["model"] for r in self._results)

        lines = [
            f"──────────────────────────────",
            f"  验收结果报告",
            f"──────────────────────────────",
            f"  总图片数: {total} 张",
            f"  模型与一致: {correct} 张",
            f"  不一致: {total - correct} 张",
            f"  一致率: {acc:.1%}",
            f"──────────────────────────────",
        ]
        for g in range(NUM_CLASSES):
            lines.append(f"  {CLASS_NAMES[g]}: 标 {human_counts[g]} 张, 模型判 {model_counts[g]} 张")

        msg = "\n".join(lines)

        result_win = tk.Toplevel(self.root)
        result_win.title("验收结果")
        result_win.geometry("660x500")
        result_win.configure(bg=C_BG)

        tk.Label(result_win, text="📊", font=("Segoe UI Emoji", 40), bg=C_BG).pack(pady=(20, 0))

        if acc >= 0.85:
            verdict = f"模型一致率 {acc:.1%} — 通过验收 ✅"
            v_color = "#4caf50"
        elif acc >= 0.70:
            verdict = f"模型一致率 {acc:.1%} — 基本合格 ⚠️"
            v_color = "#ff9800"
        else:
            verdict = f"模型一致率 {acc:.1%} — 差距较大 ❌"
            v_color = "#ef5350"

        tk.Label(result_win, text=verdict, fg=v_color, bg=C_BG,
                 font=(FONT, 20, "bold")).pack(pady=(6, 16))

        text_widget = tk.Text(result_win, bg=C_SFC, fg=C_TXT, font=(MONO, 11),
                              bd=0, padx=20, pady=16, height=16)
        text_widget.pack(fill=tk.X, padx=20)
        text_widget.insert("1.0", msg)
        text_widget.configure(state=tk.DISABLED)

        # 导出
        def _export():
            ts = time.strftime("%Y%m%d_%H%M%S")
            csv_path = BASE / f"验收报告_{ts}.csv"
            with open(csv_path, "w", encoding="utf-8-sig") as f:
                f.write("文件名,模型预测,人工标注,是否一致\n")
                for r in self._results:
                    f.write(f"{r['name']},{r['model']+1},{r['human']+1},"
                            f"{'一致' if r['model']==r['human'] else '不一致'}\n")
            # 也存 txt
            txt_path = BASE / f"验收报告_{ts}.txt"
            txt_path.write_text(msg, encoding="utf-8")
            messagebox.showinfo("导出成功", f"已保存:\n{csv_path}\n{txt_path}")

        tk.Button(result_win, text="📥  导出报告", command=_export,
                  bg="#4caf50", fg="#fff", font=(FONT, 12, "bold"), relief=tk.FLAT,
                  cursor="hand2", padx=24, pady=8).pack(pady=(16, 0))

        tk.Label(result_win, text=f"验收时间: {time.strftime('%Y-%m-%d %H:%M')}",
                 fg=C_TXT2, bg=C_BG, font=(FONT, 9)).pack(pady=10)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    DemoApp().run()
