"""诊断并修复 cache_1024 中被截断/损坏的 PNG。

背景：sync_cache.py 若被并发执行或中途打断，会残留只写了一半的 PNG，
train_3class.py 训练时加载到就会报:
    OSError: image file is truncated

流程：
  1) 扫描 cache_1024 全部 PNG，verify() 找出坏文件
  2) 对每个坏 PNG，检查源 BMP（annotated_3class）是否健康
  3) 源 BMP 健康 → 重新生成该 PNG；源 BMP 也坏 → 报告，不动它

用法:  C:/Users/LTY/anaconda3/python.exe repair_cache.py
"""
from pathlib import Path

from PIL import Image

BASE = Path(__file__).parent
ANNOTATED = BASE / "data" / "annotated_3class"
CACHE = BASE / "data" / "cache_1024"
CACHE_SIZE = 1024


def verify_img(p: Path) -> bool:
    try:
        with Image.open(p) as im:
            im.verify()
        return True
    except Exception:
        return False


def main() -> None:
    if not CACHE.exists():
        print("[WARN] 找不到 cache_1024，请先运行 sync_cache.py。")
        return

    # 1) 扫缓存
    bad_pngs: list[tuple[int, Path]] = []
    all_pngs: list[tuple[int, Path]] = []
    for cls in (1, 2, 3):
        cls_dir = CACHE / str(cls)
        if not cls_dir.exists():
            continue
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() == ".png":
                all_pngs.append((cls, p))
                if not verify_img(p):
                    bad_pngs.append((cls, p))
    print(f"缓存 PNG 总数: {len(all_pngs)}  损坏: {len(bad_pngs)}")

    # 2) 修复（源 BMP 健康才重生成；源也坏则报告）
    for cls, png in bad_pngs:
        src = ANNOTATED / str(cls) / f"{png.stem}.bmp"
        if src.exists() and verify_img(src):
            print(f"  修复 {png.name}  ← 源 BMP 健康 ({src.name})")
            png.unlink(missing_ok=True)
            im = Image.open(src).convert("RGB")
            w, h = im.size
            s = CACHE_SIZE / max(w, h)
            im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
            im.save(png)
        else:
            print(f"  [WARN] {png.name} 的源 BMP 也损坏或缺失: {src}")
            print(f"         原始图应在 data/raw/{png.stem}.bmp，请人工核对")

    if bad_pngs:
        print(f"\n[OK] 处理了 {len(bad_pngs)} 张损坏缓存，请重新运行 train_3class.py。")
    else:
        print("\n[OK] 缓存完好，没有坏文件。")


if __name__ == "__main__":
    main()
