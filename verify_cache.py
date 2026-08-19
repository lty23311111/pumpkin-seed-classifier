"""验证：512 缓存是否影响模型预测质量。

对全部图片用同一模型跑两遍预测：
  A) 全分辨率 BMP 路径（现在的读法，3072→224）
  B) 512 缓存 PNG 路径（优化后的读法，512→224）
对比两次预测的一致率。≥99.5% 视为缓存对质量无影响。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

from model_utils import build_resnet50, INFERENCE_TRANSFORM, DEVICE

BASE = Path(__file__).parent
ANNOTATED = BASE / "data" / "annotated_3class"
MODEL_PATH = BASE / "models" / "best_model_3class.pth"
CACHE_SIZE = int(os.environ.get("CACHE_SIZE", "512"))  # 缓存最长边
CACHE = BASE / "data" / f"cache_{CACHE_SIZE}"

transform = INFERENCE_TRANSFORM

# ─── 收集全部图片（按 类别/文件名 稳定排序） ───
samples: list[tuple[int, str]] = []
for cls in (1, 2, 3):
    for p in sorted((ANNOTATED / str(cls)).iterdir()):
        if p.suffix.lower() == ".bmp":
            samples.append((cls, p.name))
print(f"共 {len(samples)} 张图片")


def build_cache() -> None:
    """生成 512 缓存（保留原始宽高比，最长边=512，PNG 无损）。"""
    if (CACHE / "1").exists() and len(list((CACHE / "1").iterdir())) > 0:
        print("缓存已存在，跳过生成")
        return
    print(f"生成 512 缓存 -> {CACHE} ...")
    for cls, name in samples:
        dst_dir = CACHE / str(cls)
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{Path(name).stem}.png"
        if dst.exists():
            continue
        im = Image.open(ANNOTATED / str(cls) / name).convert("RGB")
        w, h = im.size
        s = CACHE_SIZE / max(w, h)
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
        im.save(dst)
    print("缓存生成完成")


def load_model() -> nn.Module:
    m = build_resnet50()
    m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    m = m.to(DEVICE).eval()
    return m


@torch.no_grad()
def predict(model: nn.Module, img: Image.Image) -> int:
    t = transform(img).unsqueeze(0).to(DEVICE)
    out = model(t)
    return out.argmax(1).item()


def main() -> None:
    if not MODEL_PATH.exists():
        print("未找到模型，请先训练 train_3class.py")
        sys.exit(1)

    model = load_model()
    print(f"模型已加载: {MODEL_PATH.name} (设备: {DEVICE})")

    disagree: list[str] = []
    for i, (cls, name) in enumerate(samples):
        # A) 全分辨率 BMP
        pred_a = predict(model, Image.open(ANNOTATED / str(cls) / name).convert("RGB"))
        # B) 512 缓存 PNG
        pred_b = predict(model, Image.open(CACHE / str(cls) / f"{Path(name).stem}.png").convert("RGB"))
        if pred_a != pred_b:
            disagree.append(f"{name}: 原路径→{pred_a + 1}, 缓存→{pred_b + 1}")

        if (i + 1) % 100 == 0:
            print(f"  进度 {i + 1}/{len(samples)}")

    agree = len(samples) - len(disagree)
    rate = agree / len(samples)
    print("\n" + "=" * 50)
    print(f"  总图片: {len(samples)}")
    print(f"  预测一致: {agree} ({rate:.4%})")
    print(f"  预测不一致: {len(disagree)}")
    print("=" * 50)
    for d in disagree[:20]:
        print(f"    {d}")
    if rate >= 0.995:
        print("\n[OK] 一致率 >=99.5%: 缓存对质量无影响, 可以放心使用。")
    elif rate >= 0.99:
        print("\n[WARN] 一致率 99%~99.5%: 差异极小, 缓存仍可接受。")
    else:
        print("\n[FAIL] 一致率 <99%: 缓存有可见影响, 应放弃优化或调高缓存分辨率。")


if __name__ == "__main__":
    build_cache()
    main()
