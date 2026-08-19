"""南瓜子外观质量预测 — 单张/批量推理，支持 Ensemble 投票。"""
from pathlib import Path
from collections import Counter

from model_utils import load_ensemble, CLASS_NAMES

# 全局加载
_models = load_ensemble()
_IS_ENSEMBLE = len(_models) > 1


def predict_single(image_path):
    """预测单张图片，返回 (等级名, 置信度, 各模型预测列表, 平均概率)。"""
    import torch
    from PIL import Image
    from model_utils import INFERENCE_TRANSFORM, DEVICE

    img = Image.open(image_path).convert("RGB")
    tensor = INFERENCE_TRANSFORM(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        all_preds = []
        all_probs = []
        for m in _models:
            out = m(tensor)
            all_preds.append(out.argmax(1).item())
            all_probs.append(torch.softmax(out, dim=1)[0].cpu().tolist())

    avg_probs = [sum(probs[i] for probs in all_probs) / len(all_probs)
                 for i in range(NUM_CLASSES)]
    pred = max(range(NUM_CLASSES), key=avg_probs.__getitem__)
    return CLASS_NAMES[pred], avg_probs[pred], all_preds, avg_probs


def predict_folder(folder_path):
    """预测整个文件夹。"""
    folder = Path(folder_path)
    images = sorted(list(folder.glob("*.bmp")) + list(folder.glob("*.jpg")) + list(folder.glob("*.png")))
    if not images:
        print(f"文件夹 {folder} 中没有图片文件")
        return

    counts = Counter()
    for img in images:
        grade, conf, _, _ = predict_single(img)
        counts[grade] += 1
        mode_str = f"[Ensemble {len(_models)}票]" if _IS_ENSEMBLE else ""
        print(f"  {img.name} → {grade}（置信度 {conf:.2%}）{mode_str}")

    total = len(images)
    print(f"\n  总计: {total} 张")
    for name in CLASS_NAMES:
        print(f"    {name}: {counts.get(name, 0)} 张 ({counts.get(name, 0)/total*100:.1f}%)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法:")
        print("  python predict.py <图片路径>          ← 预测单张")
        print("  python predict.py <文件夹路径>        ← 预测整个文件夹")
        sys.exit(1)

    target = Path(sys.argv[1])
    if not target.exists():
        print(f"路径不存在: {target}")
        sys.exit(1)

    if target.is_dir():
        print(f"批量预测: {target}\n")
        predict_folder(target)
    else:
        grade, conf, preds, probs = predict_single(target)
        mode_str = f" [Ensemble {len(_models)}票: {dict(Counter(preds))}]" if _IS_ENSEMBLE else ""
        print(f"{target.name} → {grade}（置信度 {conf:.2%}）{mode_str}")
        if _IS_ENSEMBLE:
            print(f"  各模型投票: {preds}")
