"""同步缓存：删除旧 cache_1024，按当前 annotated_3class 标注重新生成。

在 review_3class.py 改标之后、train_3class.py 训练之前运行，
否则训练读到的还是旧标签（缓存与标注脱节）。

用法:  C:/Users/LTY/anaconda3/python.exe sync_cache.py
"""
from pathlib import Path
import shutil

from PIL import Image

BASE = Path(__file__).parent
ANNOTATED = BASE / "data" / "annotated_3class"
CACHE = BASE / "data" / "cache_1024"
CACHE_SIZE = 1024  # 与 train_3class.py 保持一致
LOCK = BASE / ".sync_cache.lock"  # 防并发：两个实例同时跑会把缓存写坏


def collect_samples() -> list[tuple[int, Path]]:
    samples: list[tuple[int, Path]] = []
    for cls in (1, 2, 3):
        cls_dir = ANNOTATED / str(cls)
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() == ".bmp":
                samples.append((cls, p))
    return samples


def main() -> None:
    if LOCK.exists():
        print("[WARN] 检测到另一个 sync_cache.py 正在运行（或上次运行被中断）。")
        print("       确认没有其他窗口在跑后，删除文件 .sync_cache.lock 再试。")
        return
    LOCK.touch()
    try:
        rebuild()
    finally:
        LOCK.unlink(missing_ok=True)


def rebuild() -> None:
    samples = collect_samples()
    if not samples:
        print("[WARN] annotated_3class 里没有找到任何 .bmp，请先标注。")
        return

    counts = {c: sum(1 for s in samples if s[0] == c) for c in (1, 2, 3)}
    print(f"当前标注: 一级 {counts[1]}  二级 {counts[2]}  三级 {counts[3]}  合计 {len(samples)}")

    if CACHE.exists():
        shutil.rmtree(CACHE)
        print("已删除旧 cache_1024")

    n = 0
    for cls, src in samples:
        dst_dir = CACHE / str(cls)
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{src.stem}.png"
        im = Image.open(src).convert("RGB")
        w, h = im.size
        s = CACHE_SIZE / max(w, h)
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
        im.save(dst)
        n += 1
        if n % 100 == 0:
            print(f"  进度 {n}/{len(samples)}")

    print(f"[OK] 缓存已同步: cache_1024 ({n} 张)。可以运行 train_3class.py 了。")


if __name__ == "__main__":
    main()
