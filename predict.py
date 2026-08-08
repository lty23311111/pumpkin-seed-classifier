"""南瓜子外观质量预测 — 单张/批量推理，支持 Ensemble 投票。"""
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models, transforms

BASE = Path(__file__).parent
MODEL_DIR = BASE / "models"
NUM_CLASSES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["一级·完好", "二级·轻微瑕疵", "三级·明显瑕疵"]

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _load_models():
    """新高模型必入 + 旧模型补充投票。"""
    model_list = []
    loaded = set()

    def _add(p):
        if p.exists() and str(p) not in loaded:
            loaded.add(str(p))
            m = models.resnet50()
            m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(2048, NUM_CLASSES))
            m.load_state_dict(torch.load(p, map_location=DEVICE, weights_only=True))
            m = m.to(DEVICE)
            m.eval()
            model_list.append(m)

    # 新高模型必入
    _add(MODEL_DIR / "best_model_3class.pth")

    # K 折模型补充
    for fp in sorted(MODEL_DIR.glob("fold_*.pth")):
        _add(fp)

    # Top-3 补充
    for p in [MODEL_DIR / "best_model_3class_2.pth",
              MODEL_DIR / "best_model_3class_3.pth"]:
        _add(p)

    if not model_list:
        raise FileNotFoundError("未找到模型文件，请先运行 train_3class.py")

    if len(model_list) > 1:
        print(f"  已加载 {len(model_list)} 个模型 (投票)")
    else:
        print("  已加载单模型")
    return model_list


# 全局加载
_models = _load_models()
_IS_ENSEMBLE = len(_models) > 1


def predict_single(image_path):
    """预测单张图片，返回 (等级名, 置信度, 各模型预测列表, 平均概率)。"""
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    tensor = _transform(img).unsqueeze(0).to(DEVICE)
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

    from collections import Counter
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
        mode_str = f" [Ensemble {len(_models)}票: {dict(__import__('collections').Counter(preds))}]" if _IS_ENSEMBLE else ""
        print(f"{target.name} → {grade}（置信度 {conf:.2%}）{mode_str}")
        if _IS_ENSEMBLE:
            print(f"  各模型投票: {preds}")
