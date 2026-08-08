"""南瓜子外观质量分类 — 现代 GUI。支持拖拽、单张和批量预测。"""
from __future__ import annotations

import os
import threading
import time
import csv
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageTk
from torchvision import models, transforms
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD

# ═══════════════════════════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════════════════════════
BASE = Path(__file__).parent
MODEL_PATH = BASE / "models" / "best_model_3class.pth"
NUM_CLASSES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["一级 · 完好", "二级 · 轻微瑕疵", "三级 · 明显瑕疵"]
CLASS_TAGS  = ["一级", "二级", "三级"]
CLASS_ICONS = ["✓", "△", "✕"]

C = {
    "bg":           "#1f1f1f",
    "surface":      "#2c2c2c",
    "surface2":     "#363636",
    "border":       "#3e3e3e",
    "text":         "#f0f0f0",
    "text2":        "#b0b0b0",
    "text3":        "#7a7a7a",
    "accent":       "#ff8c00",
    "accent_hover": "#ffa726",
    "green":        "#4caf50",
    "amber":        "#ff9800",
    "orange":       "#ff7043",
    "red":          "#ef5350",
    "green_bg":     "#1e3520",
    "amber_bg":     "#3d301e",
    "orange_bg":    "#3d241e",
    "red_bg":       "#3d1e1e",
}
FONT = "Microsoft YaHei UI"
MONO = "Cascadia Code"

# 加载模型：新高模型必入 + 旧模型补充投票
_models: list[nn.Module] = []
_model_source = ""

model_files: list[Path] = []

best_path = BASE / "models" / "best_model_3class.pth"
if best_path.exists():
    model_files.append(best_path)

# K 折模型补充，不重复
for fp in sorted((BASE / "models").glob("fold_*.pth")):
    if fp not in model_files:
        model_files.append(fp)

# Top-3 中未覆盖的
for p in [
    BASE / "models" / "best_model_3class_2.pth",
    BASE / "models" / "best_model_3class_3.pth",
]:
    if p.exists() and p not in model_files:
        model_files.append(p)

if len(model_files) >= 5:
    _model_source = f"投票 ×{len(model_files)}（新高 + 旧折）"
elif len(model_files) >= 2:
    _model_source = f"投票 ×{len(model_files)}"
elif model_files:
    _model_source = "单模型"
else:
    raise FileNotFoundError("未找到任何模型文件，请先运行 train_3class.py")

for mp in model_files:
    m = models.resnet50()
    m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(2048, NUM_CLASSES))
    m.load_state_dict(torch.load(mp, map_location=DEVICE, weights_only=True))
    m = m.to(DEVICE)
    m.eval()
    _models.append(m)

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict_single(image_path: str | Path) -> tuple[int, list[float]]:
    img = Image.open(image_path).convert("RGB")
    t = _transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        all_probs = []
        for m in _models:
            out = m(t)
            all_probs.append(torch.softmax(out, dim=1)[0].cpu().tolist())
    avg_probs = [(sum(p[i] for p in all_probs) / len(all_probs)) for i in range(NUM_CLASSES)]
    return max(range(NUM_CLASSES), key=avg_probs.__getitem__), avg_probs


# ═══════════════════════════════════════════════════════════════
#  主应用
# ═══════════════════════════════════════════════════════════════

class SeedClassifierApp:
    def __init__(self):
        self.root = TkinterDnD.Tk()
        self.root.title("南瓜子外观质量分类系统")
        self.root.geometry("1100x780")
        self.root.configure(bg=C["bg"])
        self.root.minsize(960, 640)
        self._photo = None
        self._history: list[dict] = []
        self._build_ui()

    # ── 辅助 ───────────────────────────────────────────────────

    def _status(self, msg: str):
        self._status_text.configure(text=msg)

    def _frame(self, parent, bg=None, **kw):
        return tk.Frame(parent, bg=bg or C["bg"], **kw)

    def _card(self, parent, **kw):
        """带圆角效果的卡片——用浅色 Frame 模拟。"""
        return tk.Frame(parent, bg=C["surface"], bd=0, highlightthickness=0, **kw)

    def _sep(self, parent):
        tk.Frame(parent, bg=C["border"], height=1).pack(fill=tk.X, padx=20, pady=3)

    def _title_label(self, parent, text):
        tk.Label(parent, text=text, fg=C["text3"], bg=C["surface"],
                 font=(FONT, 10, "bold")).pack(anchor=tk.W)

    def _accent_btn(self, parent, text, command):
        b = tk.Button(parent, text=text, command=command,
                      bg=C["accent"], fg="#fff", activebackground=C["accent_hover"],
                      activeforeground="#fff", font=(FONT, 11, "bold"),
                      relief=tk.FLAT, cursor="hand2", bd=0, padx=20, pady=8)
        b.bind("<Enter>", lambda e, btn=b: btn.configure(bg=C["accent_hover"]))
        b.bind("<Leave>", lambda e, btn=b: btn.configure(bg=C["accent"]))
        return b

    def _secondary_btn(self, parent, text, command):
        b = tk.Button(parent, text=text, command=command,
                      bg=C["surface2"], fg=C["text"], activebackground="#444",
                      activeforeground=C["text"], font=(FONT, 10),
                      relief=tk.FLAT, cursor="hand2", bd=0, padx=16, pady=6)
        b.bind("<Enter>", lambda e, btn=b: btn.configure(bg="#444"))
        b.bind("<Leave>", lambda e, btn=b: btn.configure(bg=C["surface2"]))
        return b

    # ── UI 构建 ────────────────────────────────────────────────

    def _build_ui(self):
        self._build_titlebar()
        self._build_body()
        self._build_statusbar()

    def _build_titlebar(self):
        bar = tk.Frame(self.root, bg=C["surface"], height=52)
        bar.pack(side=tk.TOP, fill=tk.X)
        bar.pack_propagate(False)

        left = tk.Frame(bar, bg=C["surface"])
        left.pack(side=tk.LEFT, padx=(20, 0))
        tk.Label(left, text="🎃", font=("Segoe UI Emoji", 18), bg=C["surface"]).pack(side=tk.LEFT)
        tk.Label(left, text="  南瓜子外观质量分类", fg=C["text"], bg=C["surface"],
                 font=(FONT, 15, "bold")).pack(side=tk.LEFT, pady=12)

        right = tk.Frame(bar, bg=C["surface"])
        right.pack(side=tk.RIGHT, padx=20)
        badge = f"ResNet-50  ·  3 分类  ·  GPU  ·  {_model_source}"
        tk.Label(right, text=badge, fg=C["accent"],
                 bg=C["surface"], font=(FONT, 9)).pack(side=tk.RIGHT, pady=16)

    def _build_body(self):
        body = tk.Frame(self.root, bg=C["bg"])
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=16, pady=(14, 6))

        # ── 左侧：图片预览 ──
        left_panel = tk.Frame(body, bg=C["surface"], width=560)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left_panel.pack_propagate(False)

        self.drop_area = tk.Frame(left_panel, bg="#252525", cursor="hand2")
        self.drop_area.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.drop_area.drop_target_register(DND_FILES)
        self.drop_area.dnd_bind("<<Drop>>", self._on_drop)
        self.drop_area.bind("<Button-1>", lambda e: self._choose_single())

        # 占位引导
        self._placeholder = tk.Frame(self.drop_area, bg="#252525")
        self._placeholder.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        self._ph_icon = tk.Label(self._placeholder, text="📷", font=("Segoe UI Emoji", 52),
                                 bg="#252525")
        self._ph_icon.pack()
        self._ph_label = tk.Label(self._placeholder, text="拖拽图片到此处", fg="#666",
                                  bg="#252525", font=(FONT, 16, "bold"))
        self._ph_label.pack(pady=(10, 2))
        self._ph_hint = tk.Label(self._placeholder, text="或点击选择文件", fg="#4a4a4a",
                                 bg="#252525", font=(FONT, 11))
        self._ph_hint.pack()
        self._ph_types = tk.Label(self._placeholder, text="BMP · JPG · PNG", fg="#3a3a3a",
                                  bg="#252525", font=(FONT, 9))
        self._ph_types.pack(pady=(10, 0))
        for w in (self._ph_icon, self._ph_label, self._ph_hint, self._ph_types):
            w.bind("<Button-1>", lambda e: self._choose_single())

        self._img_label = tk.Label(self.drop_area, bg="#252525", cursor="hand2")
        self._img_label.bind("<Button-1>", lambda e: self._choose_single())

        # ── 右侧：信息面板 ──
        right_panel = tk.Frame(body, bg=C["bg"], width=500)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(14, 0))
        right_panel.pack_propagate(False)

        # 结果卡片
        result_card = self._card(right_panel)
        result_card.pack(fill=tk.X, pady=(0, 8))

        ri = tk.Frame(result_card, bg=C["surface"])
        ri.pack(padx=20, pady=18, fill=tk.X)

        self._title_label(ri, "预测结果")

        # 等级显示区
        self._grade_frame = tk.Frame(ri, bg=C["green_bg"])
        self._grade_frame.pack(fill=tk.X, pady=(10, 4))
        self._grade_icon = tk.Label(self._grade_frame, text="", fg=C["green"],
                                    bg=C["green_bg"], font=("Segoe UI Emoji", 20))
        self._grade_icon.pack(side=tk.LEFT, padx=(20, 8), pady=14)
        self._grade_label = tk.Label(self._grade_frame, text="等待图片...", fg=C["green"],
                                     bg=C["green_bg"], font=(FONT, 24, "bold"))
        self._grade_label.pack(side=tk.LEFT, pady=14)

        self._conf_label = tk.Label(ri, text="", fg=C["text2"], bg=C["surface"],
                                    font=(FONT, 13))
        self._conf_label.pack(anchor=tk.W, pady=(6, 0))

        self._sep(right_panel)

        # 概率卡片
        prob_card = self._card(right_panel)
        prob_card.pack(fill=tk.X, pady=(8, 8))
        pi = tk.Frame(prob_card, bg=C["surface"])
        pi.pack(padx=20, pady=16, fill=tk.X)

        self._title_label(pi, "各类概率")

        self._prob_bars: dict[int, tk.Canvas] = {}
        self._prob_cfg = {0: C["green"], 1: C["amber"], 2: C["red"]}
        for i, name in enumerate(CLASS_NAMES):
            row = tk.Frame(pi, bg=C["surface"])
            row.pack(fill=tk.X, pady=5)
            tk.Label(row, text=f" {CLASS_ICONS[i]}  {name}", fg=C["text2"],
                     bg=C["surface"], width=17, anchor=tk.W,
                     font=(FONT, 11)).pack(side=tk.LEFT)
            cv = tk.Canvas(row, bg="#3a3a3a", height=28, bd=0, highlightthickness=0)
            cv.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self._prob_bars[i] = cv

        self._sep(right_panel)

        # 按钮
        btn_area = tk.Frame(right_panel, bg=C["bg"])
        btn_area.pack(fill=tk.X, pady=(8, 0))
        self._accent_btn(btn_area, "📁  批量预测文件夹", self._choose_folder).pack(fill=tk.X)

    def _build_statusbar(self):
        bar = tk.Frame(self.root, bg=C["surface"], height=30)
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        bar.pack_propagate(False)

        self._status_text = tk.Label(bar, text="就绪 · 模型已加载", fg=C["text3"],
                                     bg=C["surface"], anchor=tk.W, font=(FONT, 9))
        self._status_text.pack(side=tk.LEFT, padx=18, pady=5)

        self._stat_right = tk.Label(bar, text="", fg=C["text3"], bg=C["surface"],
                                    font=(FONT, 9))
        self._stat_right.pack(side=tk.RIGHT, padx=18, pady=5)

    # ── 绘制 ───────────────────────────────────────────────────

    def _draw_grade(self, pred: int):
        colors = {0: (C["green_bg"], C["green"]),
                  1: (C["amber_bg"], C["amber"]),
                  2: (C["red_bg"], C["red"])}
        bg_c, fg_c = colors[pred]
        self._grade_frame.configure(bg=bg_c)
        self._grade_icon.configure(text=CLASS_ICONS[pred], fg=fg_c, bg=bg_c)
        self._grade_label.configure(text=CLASS_NAMES[pred], fg=fg_c, bg=bg_c)

    def _draw_bars(self, probs: list[float]):
        for i in range(NUM_CLASSES):
            cv = self._prob_bars[i]
            cv.delete("all")
            # 等 Canvas 渲染后再画
            w = cv.winfo_width()
            if w < 4:
                w = 300
            bar_w = int(probs[i] * w)
            if bar_w > 0:
                cv.create_rectangle(0, 0, bar_w, 30, fill=self._prob_cfg[i], outline="")
            cv.create_text(14, 15, text=f"{probs[i]:.1%}", anchor=tk.W,
                           fill="#fff", font=(MONO, 11, "bold"))

    # ── 核心 ───────────────────────────────────────────────────

    def _show_image(self, path: Path):
        img = Image.open(path).convert("RGB")
        max_w, max_h = 540, 450
        w, h = img.size
        s = min(max_w / w, max_h / h, 1.0)
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)
        self._img_label.configure(image=self._photo, text="", cursor="hand2")
        self._img_label.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self._placeholder.place_forget()

    def _show_zoom(self, path: str | Path):
        """在独立窗口内放大显示图片（不依赖系统关联程序）。"""
        win = tk.Toplevel(self.root)
        win.title(Path(path).name)
        win.configure(bg=C["bg"])

        img = Image.open(path).convert("RGB")
        max_w, max_h = 1000, 760
        w, h = img.size
        s = min(max_w / w, max_h / h, 1.0)
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)

        photo = ImageTk.PhotoImage(img)
        lbl = tk.Label(win, image=photo, bg=C["bg"])
        lbl.image = photo  # 防止被垃圾回收
        lbl.pack(padx=8, pady=8)

        # 窗口居中于主界面
        win.update_idletasks()
        x = self.root.winfo_x() + max(0, (self.root.winfo_width() - win.winfo_width()) // 2)
        y = self.root.winfo_y() + max(0, (self.root.winfo_height() - win.winfo_height()) // 2)
        win.geometry(f"+{x}+{y}")

    def _infer(self, path: str | Path):
        path = Path(str(path).strip("{}").replace("/", "\\"))
        if not path.exists() or not path.is_file():
            return
        if path.suffix.lower() not in (".bmp", ".jpg", ".jpeg", ".png"):
            return

        self._show_image(path)
        self._status("⏳ 分析中: " + path.name)

        def _run():
            pred, probs = predict_single(path)
            self.root.after(0, lambda: self._display(pred, probs, path))

        threading.Thread(target=_run, daemon=True).start()

    def _display(self, pred: int, probs: list[float], path: Path):
        conf = probs[pred]
        self._draw_grade(pred)
        self._conf_label.configure(text=f"置信度  {conf:.1%}")
        self._draw_bars(probs)
        self._status(f"{path.name}  ·  {CLASS_TAGS[pred]}  ·  {conf:.1%}")

        self._history.insert(0, {
            "name": path.name, "grade": pred, "conf": conf,
            "time": time.strftime("%H:%M:%S")})
        if len(self._history) > 100:
            self._history.pop()
        self._stat_right.configure(text=f"已检测 {len(self._history)} 张")

    # ── 事件 ───────────────────────────────────────────────────

    def _on_drop(self, event):
        files = self.root.tk.splitlist(event.data)
        if files:
            self._infer(files[0])

    def _choose_single(self):
        path = filedialog.askopenfilename(
            title="选择南瓜子图片",
            filetypes=[("图片文件", "*.bmp *.jpg *.jpeg *.png"), ("所有文件", "*.*")])
        if path:
            self._infer(path)

    def _choose_folder(self):
        folder = filedialog.askdirectory(title="选择图片文件夹")
        if not folder:
            return
        folder = Path(folder)
        images = sorted(
            list(folder.glob("*.bmp")) + list(folder.glob("*.jpg")) +
            list(folder.glob("*.jpeg")) + list(folder.glob("*.png")))
        if not images:
            messagebox.showinfo("提示", "文件夹中没有图片文件")
            return

        self._status(f"⏳ 批量预测中... 0/{len(images)}")

        # ── 批量结果弹窗 ──
        win = tk.Toplevel(self.root)
        win.title(f"批量预测  ·  {len(images)} 张")
        win.geometry("920x600")
        win.configure(bg=C["bg"])
        win.minsize(700, 400)

        # 统计栏
        stat_bar = tk.Frame(win, bg=C["surface"])
        stat_bar.pack(fill=tk.X)
        si = tk.Frame(stat_bar, bg=C["surface"])
        si.pack(padx=20, pady=14, fill=tk.X)

        stat_main = tk.Label(si, text=f"共 {len(images)} 张  ·  处理中...",
                             fg=C["text"], bg=C["surface"], font=(FONT, 14, "bold"))
        stat_main.pack(side=tk.LEFT)
        stat_sub = tk.Label(si, text="", fg=C["text2"], bg=C["surface"], font=(FONT, 10))
        stat_sub.pack(side=tk.LEFT, padx=16)

        prog = ttk.Progressbar(si, length=180, mode="determinate")
        prog.pack(side=tk.RIGHT)

        # 表格
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=C["bg"])
        style.configure("Treeview", background=C["surface2"], foreground=C["text"],
                        fieldbackground=C["surface2"], rowheight=30,
                        font=(FONT, 10), borderwidth=0)
        style.configure("Treeview.Heading", background=C["surface"],
                        foreground=C["text3"], font=(FONT, 10, "bold"),
                        borderwidth=1, relief="solid")
        style.map("Treeview", background=[("selected", "#3a3a3a")])

        tree_frame = tk.Frame(win, bg=C["bg"])
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        cols = ("序号", "文件名", "等级", "置信度", "一级概率", "二级概率", "三级概率")
        widths = [50, 280, 90, 80, 80, 80, 80]
        tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=18)
        for c, w in zip(cols, widths):
            tree.heading(c, text=c)
            tree.column(c, width=w, anchor=tk.CENTER)
        tree.column("文件名", anchor=tk.W)

        sb = ttk.Scrollbar(tree, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        tree.tag_configure("g0", foreground=C["green"])
        tree.tag_configure("g1",  foreground=C["amber"])
        tree.tag_configure("g2",  foreground=C["red"])

        # 底部按钮
        btn_bar = tk.Frame(win, bg=C["bg"])
        btn_bar.pack(fill=tk.X, padx=10, pady=(0, 8))
        counts: Counter[int] = Counter()
        img_paths: dict[str, str] = {}  # tree item id → 完整路径

        def _open_image(event=None):
            """双击行 → 打开图片预览（窗口内显示，保证可见）。"""
            item = None
            # 1) 当前选中的行
            sel = tree.selection()
            if sel:
                item = sel[0]
            # 2) 光标所在行（ttk 双击有时先于「选中行」生效）
            if not item and event is not None:
                row = tree.identify_row(event.y)
                if row:
                    item = row
                    tree.selection_set(row)
            # 3) 焦点行兜底
            if not item:
                item = tree.focus()
            if not item:
                return

            p = img_paths.get(item)
            if not p:
                return

            path = Path(p)
            if not path.exists():
                messagebox.showwarning(
                    "文件不存在",
                    f"图片已被移动或删除，无法打开：\n{p}\n\n"
                    f"可能是在重标注时被移到了其他类别文件夹。")
                return

            # 窗口内预览：不依赖系统文件关联，且必定显示在最上层
            try:
                self._show_zoom(path)
            except Exception as e:
                messagebox.showerror("打开失败", f"无法打开图片：\n{p}\n\n{e}")

        tree.bind("<Double-1>", _open_image)

        # 右键菜单：复制文件名 / 打开图片
        _right_click_menu = tk.Menu(win, tearoff=0, bg=C["surface2"], fg=C["text"],
                                     font=(FONT, 10))
        def _on_right_click(event):
            sel = tree.selection()
            if sel:
                p = img_paths.get(sel[0], "")
                _right_click_menu.delete(0, tk.END)
                _right_click_menu.add_command(
                    label="📂 打开图片", command=lambda: os.startfile(p) if p else None)
                _right_click_menu.add_command(
                    label="📋 复制路径", command=lambda: self.root.clipboard_append(p) if p else None)
                _right_click_menu.add_command(
                    label="📋 复制文件名",
                    command=lambda: self.root.clipboard_append(Path(p).name) if p else None)
                _right_click_menu.tk_popup(event.x_root, event.y_root)
        tree.bind("<Button-3>", _on_right_click)

        hint_label = tk.Label(btn_bar, text="💡 双击打开图片  ·  右键复制路径", fg=C["text3"],
                              bg=C["bg"], font=(FONT, 9))
        hint_label.pack(side=tk.LEFT, padx=(10, 0))

        def _export():
            csv_path = folder / f"预测结果_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(cols)
                for item in tree.get_children():
                    writer.writerow(tree.item(item)["values"])
            messagebox.showinfo("导出成功", f"已保存至:\n{csv_path}")

        self._secondary_btn(btn_bar, "📥  导出 CSV", _export).pack(side=tk.RIGHT)

        def _append_result(i, img, pred, probs):
            """Tkinter 控件只能由主线程更新。"""
            tag = f"g{pred}"
            item_id = tree.insert("", tk.END, values=(
                i + 1, img.name, CLASS_TAGS[pred], f"{probs[pred]:.1%}",
                f"{probs[0]:.1%}", f"{probs[1]:.1%}", f"{probs[2]:.1%}"),
                tags=(tag,))
            img_paths[item_id] = str(img)

        def _batch_run():
            for i, img in enumerate(images):
                pred, probs = predict_single(img)
                counts[pred] += 1
                self.root.after(0, _append_result, i, img, pred, probs)
                self.root.after(0, lambda c=i+1: [
                    prog.configure(value=100 * c / len(images)),
                    stat_main.configure(text=f"共 {len(images)} 张  ·  {c}/{len(images)}")])

            parts = "  ╱  ".join(
                f"{CLASS_TAGS[i]} {counts[i]} 张 ({counts[i] / len(images) * 100:.1f}%)"
                for i in range(NUM_CLASSES))
            self.root.after(0, lambda: [
                stat_main.configure(text=f"✅  {len(images)} 张全部完成"),
                stat_sub.configure(text=parts),
                self._status(f"批量完成 · {len(images)} 张")])

        threading.Thread(target=_batch_run, daemon=True).start()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    SeedClassifierApp().run()
