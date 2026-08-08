"""南瓜子外观质量 — 三分类标注工具。

把 data/raw/ 里的图片逐张标注到 data/annotated_3class/{1,2,3}/。
- 按 1/2/3 键归类（自动复制到对应类别文件夹 + 跳到下一张未标注）
- ← → 翻页；Backspace 回退；Esc 退出
- 断点续标：已经标注过的图片自动跳过，可以随时关闭再继续
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

BASE = Path(__file__).parent
RAW = BASE / "data" / "raw"
ANNOTATED = BASE / "data" / "annotated_3class"

NUM_CLASSES = 3
CLASS_NAMES = ["一级 · 完好", "二级 · 轻微瑕疵", "三级 · 明显瑕疵"]
CLASS_SHORT = ["一级", "二级", "三级"]
CLASS_COLORS = ["#4caf50", "#ff9800", "#ef5350"]
CLASS_BG = ["#1e3520", "#3d301e", "#3d1e1e"]

C_BG = "#1f1f1f"
C_SFC = "#2c2c2c"
C_TXT = "#f0f0f0"
C_TXT2 = "#b0b0b0"
C_TXT3 = "#7a7a7a"
FONT = "Microsoft YaHei UI"
MONO = "Cascadia Code"

IMG_MAX_W, IMG_MAX_H = 1160, 740


def list_images() -> list[Path]:
    """列出 raw 里所有图片（按文件名排序，稳定顺序）。"""
    exts = (".bmp", ".jpg", ".jpeg", ".png")
    return sorted(p for p in RAW.iterdir() if p.suffix.lower() in exts)


def existing_label(img_name: str) -> int | None:
    """图片是否已经标注过，返回类别(0/1/2)或 None。"""
    for i in range(NUM_CLASSES):
        if (ANNOTATED / str(i + 1) / img_name).exists():
            return i
    return None


class Annotator:
    def __init__(self, files: list[Path]):
        self.files = files
        self.total = len(files)
        self.idx = 0
        self.photo = None

        self.root = tk.Tk()
        self.root.title("南瓜子三分类标注")
        self.root.geometry("1280x860")
        self.root.configure(bg=C_BG)
        self.root.minsize(1024, 700)

        self._build_ui()
        self.show()
        self._update_stats()

    # ── UI ────────────────────────────────────────────────────

    def _build_ui(self):
        # 顶栏
        bar = tk.Frame(self.root, bg=C_SFC, height=50)
        bar.pack(fill=tk.X)
        bar.pack_propagate(False)
        tk.Label(bar, text="🌻", font=("Segoe UI Emoji", 16), bg=C_SFC).pack(side=tk.LEFT, padx=(18, 6))
        tk.Label(bar, text="南瓜子三分类标注", fg=C_TXT, bg=C_SFC,
                 font=(FONT, 14, "bold")).pack(side=tk.LEFT, pady=12)
        self.pos_label = tk.Label(bar, text="", fg=C_TXT2, bg=C_SFC, font=(FONT, 11))
        self.pos_label.pack(side=tk.LEFT, padx=24)
        self.stats_label = tk.Label(bar, text="", fg="#ff8c00", bg=C_SFC, font=(FONT, 11, "bold"))
        self.stats_label.pack(side=tk.RIGHT, padx=20)

        # 主体：左图右信息
        body = tk.Frame(self.root, bg=C_BG)
        body.pack(fill=tk.BOTH, expand=True, padx=14, pady=(12, 6))

        left = tk.Frame(body, bg="#252525")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.img_label = tk.Label(left, bg="#252525")
        self.img_label.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        right = tk.Frame(body, bg=C_SFC, width=380)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(12, 0))
        right.pack_propagate(False)
        inner = tk.Frame(right, bg=C_SFC)
        inner.pack(fill=tk.BOTH, expand=True, padx=22, pady=20)

        tk.Label(inner, text="文件", fg=C_TXT3, bg=C_SFC, font=(FONT, 10, "bold")).pack(anchor=tk.W)
        self.file_label = tk.Label(inner, text="", fg=C_TXT2, bg=C_SFC,
                                   font=(MONO, 10), wraplength=330, justify=tk.LEFT)
        self.file_label.pack(anchor=tk.W, pady=(4, 16))

        tk.Label(inner, text="当前标注", fg=C_TXT3, bg=C_SFC, font=(FONT, 10, "bold")).pack(anchor=tk.W)
        self.grade_box = tk.Frame(inner, bg=C_BG)
        self.grade_box.pack(fill=tk.X, pady=(4, 16))
        self.grade_text = tk.Label(self.grade_box, text="未标注", fg=C_TXT3, bg=C_BG,
                                   font=(FONT, 20, "bold"))
        self.grade_text.pack(padx=16, pady=14)

        tk.Label(inner, text="归类（快捷键）", fg=C_TXT3, bg=C_SFC, font=(FONT, 10, "bold")).pack(anchor=tk.W)
        self._grade_btns = []
        for g in range(NUM_CLASSES):
            b = tk.Button(inner, text=f"  [{g + 1}]  {CLASS_NAMES[g]}",
                          command=lambda g0=g: self.assign(g0),
                          bg=CLASS_COLORS[g], fg="#fff",
                          font=(FONT, 12, "bold"), relief=tk.FLAT, cursor="hand2",
                          padx=12, pady=10)
            b.pack(fill=tk.X, pady=4)
            b.bind("<Enter>", lambda e, btn=b, c=g: btn.configure(bg=self._lighter(CLASS_COLORS[c])))
            b.bind("<Leave>", lambda e, btn=b, c=g: btn.configure(bg=CLASS_COLORS[c]))
            self._grade_btns.append(b)

        tk.Frame(inner, bg=C_SFC, height=8).pack()

        tk.Label(inner, text="各分类进度", fg=C_TXT3, bg=C_SFC, font=(FONT, 10, "bold")).pack(anchor=tk.W)
        self.count_labels = []
        for g in range(NUM_CLASSES):
            lb = tk.Label(inner, text="", fg=CLASS_COLORS[g], bg=C_SFC, font=(FONT, 11))
            lb.pack(anchor=tk.W, pady=1)
            self.count_labels.append(lb)

        tk.Frame(inner, bg=C_SFC, height=12).pack()

        # 底部提示
        hint = tk.Frame(self.root, bg=C_SFC, height=44)
        hint.pack(fill=tk.X, side=tk.BOTTOM)
        hint.pack_propagate(False)
        tk.Label(hint, text="1/2/3 归类 · ←→ 翻页 · Backspace 返回 · Esc 退出",
                 fg=C_TXT3, bg=C_SFC, font=(FONT, 10)).pack(pady=11)

        # 键盘
        for g in range(NUM_CLASSES):
            self.root.bind(str(g + 1), lambda e, g0=g: self.assign(g0))
        self.root.bind("<Left>", lambda _: self.go(-1))
        self.root.bind("<Right>", lambda _: self.go(1))
        self.root.bind("<BackSpace>", lambda _: self.go(-1))
        self.root.bind("<Escape>", lambda _: self.quit())

        # 全局进度条（顶部右侧）
        self._progress = ttk.Progressbar(bar, length=180, mode="determinate")
        self._progress.pack(side=tk.RIGHT, padx=(0, 20))

    @staticmethod
    def _lighter(hex_color):
        r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
        r, g, b = min(255, r + 36), min(255, g + 36), min(255, b + 36)
        return f"#{r:02x}{g:02x}{b:02x}"

    # ── 核心 ──────────────────────────────────────────────────

    def show(self):
        path = self.files[self.idx]

        # 图片
        img = Image.open(path).convert("RGB")
        w, h = img.size
        s = min(IMG_MAX_W / w, IMG_MAX_H / h, 1.0)
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(img)
        self.img_label.configure(image=self.photo)

        # 文件与当前标注
        self.file_label.configure(text=path.name)
        label = existing_label(path.name)
        color = CLASS_COLORS[label] if label is not None else C_TXT3
        text = CLASS_NAMES[label] if label is not None else "未标注"
        self.grade_box.configure(bg=CLASS_BG[label] if label is not None else C_BG)
        self.grade_text.configure(text=text, fg=color, bg=CLASS_BG[label] if label is not None else C_BG)

        self.pos_label.configure(text=f"{self.idx + 1} / {self.total}")

    def assign(self, grade: int):
        path = self.files[self.idx]
        old = existing_label(path.name)
        if old == grade:
            # 已经是该类别，直接跳到下一张未标注
            self._next_unlabeled()
            return
        if old is not None:
            (ANNOTATED / str(old + 1) / path.name).unlink(missing_ok=True)
        dst = ANNOTATED / str(grade + 1) / path.name
        shutil.copy2(path, dst)
        self._update_stats()
        self._next_unlabeled()

    def _next_unlabeled(self):
        """跳到下一张未标注的图片；没有则停在当前。"""
        for _ in range(self.total):
            self.idx = (self.idx + 1) % self.total
            if existing_label(self.files[self.idx].name) is None:
                break
        self.show()
        self._update_stats()

    def go(self, step: int):
        self.idx = (self.idx + step) % self.total
        self.show()
        self._update_stats()

    def quit(self):
        self.root.destroy()

    def _update_stats(self):
        labeled = 0
        counts = [0] * NUM_CLASSES
        for g in range(NUM_CLASSES):
            n = len(list((ANNOTATED / str(g + 1)).glob("*")))
            counts[g] = n
            labeled += n
        remaining = self.total - labeled
        self.stats_label.configure(text=f"已标 {labeled}  剩余 {remaining}")
        self._progress.configure(value=labeled, maximum=max(self.total, 1))
        for g in range(NUM_CLASSES):
            self.count_labels[g].configure(
                text=f"●  {CLASS_NAMES[g]}：{counts[g]} 张  ({counts[g] / self.total * 100:.1f}%)")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    files = list_images()
    if not files:
        print("data/raw/ 里没有图片，请先放入原始图片。")
        sys.exit(1)
    print(f"共 {len(files)} 张图片，打开标注窗口...")
    Annotator(files).run()
    print("标注完成。")
