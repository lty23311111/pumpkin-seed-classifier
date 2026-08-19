"""公共模型加载与软投票推理。所有推理入口统一调用此模块。

集中管理：
  - ResNet-50 模型构建
  - Ensemble 模型加载（best + fold + Top-3）
  - 软投票推理
  - 推理 / 训练用 transforms
  - 类别名、设备等全局常量
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms

# ═══════════════════════════════════════════════════════════════
#  全局常量
# ═══════════════════════════════════════════════════════════════

BASE = Path(__file__).parent
MODEL_DIR = BASE / "models"
NUM_CLASSES = 3
IMG_SIZE = 224

CLASS_NAMES = ["一级·完好", "二级·轻微瑕疵", "三级·明显瑕疵"]
CLASS_NAMES_SPACED = ["一级 · 完好", "二级 · 轻微瑕疵", "三级 · 明显瑕疵"]
CLASS_TAGS = ["一级", "二级", "三级"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ═══════════════════════════════════════════════════════════════
#  Transforms
# ═══════════════════════════════════════════════════════════════

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
])

VAL_TRANSFORM = INFERENCE_TRANSFORM  # 验证与推理用同一 transform


# ═══════════════════════════════════════════════════════════════
#  模型构建
# ═══════════════════════════════════════════════════════════════

def build_resnet50(num_classes: int = NUM_CLASSES,
                   pretrained: bool = False) -> nn.Module:
    """构建 ResNet-50。pretrained=True 时加载 ImageNet1K_V2 预训练权重。"""
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    m = models.resnet50(weights=weights)
    m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.fc.in_features, num_classes))
    return m


# ═══════════════════════════════════════════════════════════════
#  Ensemble 加载
# ═══════════════════════════════════════════════════════════════

def load_ensemble(device: torch.device = DEVICE,
                  verbose: bool = True) -> list[nn.Module]:
    """扫描 models/ 目录，按固定优先级加载全部可用模型，返回 eval 模式列表。

    加载顺序：best → fold_1~5 → Top-2/3（去重）。
    """
    loaded: list[nn.Module] = []
    seen: set[str] = set()

    def _add(p: Path) -> None:
        key = str(p.resolve())
        if p.exists() and key not in seen:
            seen.add(key)
            m = build_resnet50()
            m.load_state_dict(torch.load(p, map_location=device, weights_only=True))
            m.to(device).eval()
            loaded.append(m)

    # 新高模型必入
    _add(MODEL_DIR / "best_model_3class.pth")

    # K 折模型补充
    for fp in sorted(MODEL_DIR.glob("fold_*.pth")):
        _add(fp)

    # Top-2/3 补充
    _add(MODEL_DIR / "best_model_3class_2.pth")
    _add(MODEL_DIR / "best_model_3class_3.pth")

    if not loaded:
        raise FileNotFoundError("未找到模型文件，请先运行 train_3class.py")

    if verbose:
        if len(loaded) > 1:
            print(f"  已加载 {len(loaded)} 个模型 (投票)")
        else:
            print("  已加载单模型")

    return loaded


def ensemble_description(n_models: int) -> str:
    """返回模型来源的简短描述文字（用于 UI badge）。"""
    if n_models >= 5:
        return f"投票 ×{n_models}（新高 + 旧折）"
    elif n_models >= 2:
        return f"投票 ×{n_models}"
    else:
        return "单模型"


# ═══════════════════════════════════════════════════════════════
#  软投票推理
# ═══════════════════════════════════════════════════════════════

def predict_soft_vote(models_list: list[nn.Module],
                      tensor: torch.Tensor,
                      device: torch.device = DEVICE) -> tuple[int, list[float]]:
    """软投票推理。

    Args:
        models_list: eval 模式的模型列表。
        tensor: 形状 (1, C, H, W) 的输入张量。
        device: 计算设备。

    Returns:
        (预测类别索引, 各类平均概率列表)
    """
    tensor = tensor.to(device)
    all_probs: list[list[float]] = []
    with torch.no_grad():
        for m in models_list:
            out = m(tensor)
            all_probs.append(torch.softmax(out, dim=1)[0].cpu().tolist())
    avg = [sum(p[i] for p in all_probs) / len(all_probs)
           for i in range(NUM_CLASSES)]
    pred = max(range(NUM_CLASSES), key=avg.__getitem__)
    return pred, avg


def predict_image(models_list: list[nn.Module],
                  image_path: str | Path,
                  device: torch.device = DEVICE) -> tuple[int, list[float]]:
    """从图片路径直接做软投票推理。

    Returns:
        (预测类别索引, 各类平均概率列表)
    """
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    tensor = INFERENCE_TRANSFORM(img).unsqueeze(0)
    return predict_soft_vote(models_list, tensor, device)
