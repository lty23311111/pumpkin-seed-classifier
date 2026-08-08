"""南瓜子外观质量分类 — 5 折交叉验证。与 train_3class.py 共用配方，评估真实水平。

以「留出验证准确率」为模型选择与评估口径（与西瓜子项目审查后一致）。
"""
from pathlib import Path
import csv
import datetime
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, models, transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from tqdm import tqdm

BASE = Path(__file__).parent
# 使用 512/1024 预处理缓存加速（verify_cache.py 验证 1024 与全分辨率一致率 99.83%）
# 若要改回全分辨率原始图，把 ANNOTATED 改回 "annotated_3class" 即可
ANNOTATED = BASE / "data" / "cache_1024"
MODEL_DIR = BASE / "models"
KFOLD_LOG    = BASE / "kfold_log.csv"
KFOLD_RECORD = MODEL_DIR / "_kfold_record_3class.json"
MODEL_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 60
LR = 0.001
IMG_SIZE = 224
NUM_CLASSES = 3
N_FOLDS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["一级·完好", "二级·轻微瑕疵", "三级·明显瑕疵"]

# ─── 数据增强（与 train_3class.py 完全一致） ───
train_transform = transforms.Compose([
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

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ─── 数据集 ───
print("加载数据集...")
full_dataset = datasets.ImageFolder(root=str(ANNOTATED))
targets = np.array(full_dataset.targets)
print(f"  类别: {full_dataset.classes}")
class_counts = [int((targets == i).sum()) for i in range(NUM_CLASSES)]
print(f"  每类数量: {class_counts}")
print(f"  总计: {len(full_dataset)} 张")

# ─── 5 折分层拆分 ───
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
fold_splits = list(skf.split(np.zeros(len(targets)), targets))
print(f"\n{'='*60}")
print(f"  5 折交叉验证：每折训练 ~{len(fold_splits[0][0])} 张，验证 ~{len(fold_splits[0][1])} 张")
print(f"{'='*60}")


def build_model():
    """每次从头构建新模型（各折独立，不共享权重）。"""
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.fc.in_features, NUM_CLASSES))
    return m.to(DEVICE)


def train_one_fold(fold_idx: int, train_idx: np.ndarray, val_idx: np.ndarray,
                   old_val_acc: float = 0.0):
    """训练一折，返回 (指标字典, 是否保存了新模型)。"""
    print(f"\n{'─'*60}")
    print(f"  📂 第 {fold_idx + 1}/{N_FOLDS} 折")
    if old_val_acc > 0:
        print(f"     旧留出验证准确率: {old_val_acc:.4f}")
    print(f"{'─'*60}")

    train_targets = targets[train_idx]

    train_ds = Subset(datasets.ImageFolder(root=str(ANNOTATED), transform=train_transform),
                      train_idx.tolist())
    val_ds = Subset(datasets.ImageFolder(root=str(ANNOTATED), transform=val_transform),
                    val_idx.tolist())

    # 均衡采样
    class_sample_counts = [int((train_targets == i).sum()) for i in range(NUM_CLASSES)]
    sample_weights = [1.0 / class_sample_counts[t] for t in train_targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_targets),
                                   replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 模型 & 优化器
    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    temp_path = MODEL_DIR / f"_temp_fold_{fold_idx}.pth"

    # ── 训练 ──
    pbar = tqdm(range(EPOCHS), desc=f"  Fold {fold_idx+1}", ncols=90)
    for epoch in pbar:
        model.train()
        train_loss, train_correct = 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()

        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)

        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item() * inputs.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), temp_path)

        pbar.set_postfix(Train=f"{train_acc:.3f}", Val=f"{val_acc:.3f}",
                         Best=f"{best_val_acc:.3f}")

    # 加载最佳权重
    model.load_state_dict(torch.load(temp_path, map_location=DEVICE))
    model.eval()
    temp_path.unlink()

    # ── 验证集评估（该折的留出数据） ──
    all_preds_v, all_labels_v = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            all_preds_v.extend(outputs.argmax(1).cpu().tolist())
            all_labels_v.extend(labels.cpu().tolist())

    fold_val_acc = (np.array(all_labels_v) == np.array(all_preds_v)).mean()
    val_report = classification_report(all_labels_v, all_preds_v,
                                       target_names=CLASS_NAMES, digits=4,
                                       output_dict=True, zero_division=0)

    # ── 全量评估（全部 600 张，与 train_3class.py 口径一致） ──
    full_loader = DataLoader(
        datasets.ImageFolder(root=str(ANNOTATED), transform=val_transform),
        batch_size=BATCH_SIZE, shuffle=False)
    all_preds_f, all_labels_f = [], []
    with torch.no_grad():
        for inputs, labels in full_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            all_preds_f.extend(outputs.argmax(1).cpu().tolist())
            all_labels_f.extend(labels.cpu().tolist())

    full_acc = (np.array(all_labels_f) == np.array(all_preds_f)).mean()
    class_f1 = {name: val_report[name]["f1-score"] for name in CLASS_NAMES}

    print(f"  留出验证准确率: {fold_val_acc:.4f}  全量准确率: {full_acc:.4f}")

    # ── 保存该折模型（只覆盖更好的） ──
    fold_path = MODEL_DIR / f"fold_{fold_idx + 1}.pth"
    if fold_val_acc > old_val_acc:
        torch.save(model.state_dict(), fold_path)
        improved = True
        print(f"  ✅ 已保存 {fold_path.name}（留出验证 {fold_val_acc:.4f} > 旧 {old_val_acc:.4f}）")
    else:
        improved = False
        print(f"  ⏭️  保留旧 {fold_path.name}（留出验证 {fold_val_acc:.4f} ≤ 旧 {old_val_acc:.4f}）")

    return {
        "fold": fold_idx + 1,
        "val_accuracy": fold_val_acc,
        "full_accuracy": full_acc,
        "macro_f1": val_report["macro avg"]["f1-score"],
        "class_f1": class_f1,
        "best_val_epoch_acc": best_val_acc,
    }, improved


# ─── 主循环 ───
# 读取上一轮的折准确率，用于比较
if KFOLD_RECORD.exists():
    old_records = json.loads(KFOLD_RECORD.read_text())
    old_fold_accs = ({r["fold"]: r["accuracy"] for r in old_records.get("folds", [])}
                     if old_records.get("metric") == "held_out_accuracy" else {})
    print(f"\n旧 K 折记录: {old_fold_accs}")
else:
    old_fold_accs = {}
    print()

results = []
improved_count = 0
start_time = datetime.datetime.now()

for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
    fold_num = fold_idx + 1
    old_acc = old_fold_accs.get(fold_num, 0.0)
    r, improved = train_one_fold(fold_idx, train_idx, val_idx, old_acc)
    if improved:
        improved_count += 1
    results.append(r)

# 更新折准确率记录（只保存更好的）
fold_accuracies = []
for r in results:
    fold_num = r["fold"]
    new_acc = r["val_accuracy"]
    old_acc = old_fold_accs.get(fold_num, 0.0)
    fold_accuracies.append({
        "fold": fold_num,
        "accuracy": max(new_acc, old_acc),
    })
KFOLD_RECORD.write_text(
    json.dumps({"metric": "held_out_accuracy", "folds": fold_accuracies},
               indent=2, ensure_ascii=False),
    encoding="utf-8")

elapsed = datetime.datetime.now() - start_time
print(f"\n共 {improved_count}/{N_FOLDS} 折被更新")

# ─── 汇总 ───
print(f"\n{'='*60}")
print(f"  5 折交叉验证结果")
print(f"{'='*60}")

full_accs = [r["full_accuracy"] for r in results]
val_accs = [r["val_accuracy"] for r in results]
macro_f1s = [r["macro_f1"] for r in results]

print(f"\n  {'折':<6} {'留出验证':>8} {'全量准确率':>10} {'宏平均 F1':>10}")
print(f"  {'─'*40}")
for r in results:
    print(f"  Fold {r['fold']:<2}  {r['val_accuracy']:>8.4f}  {r['full_accuracy']:>10.4f}  "
          f"{r['macro_f1']:>10.4f}")

print(f"\n  {'─'*40}")
print(f"  均值     {np.mean(val_accs):>8.4f}  {np.mean(full_accs):>10.4f}  "
      f"{np.mean(macro_f1s):>10.4f}")
print(f"  标准差    {np.std(val_accs):>8.4f}  {np.std(full_accs):>10.4f}  "
      f"{np.std(macro_f1s):>10.4f}")

# 每类 F1 汇总
print(f"\n  各类 F1 均值:")
for i, name in enumerate(CLASS_NAMES):
    class_f1s = [r["class_f1"][name] for r in results]
    print(f"    {name}: {np.mean(class_f1s):.4f} ± {np.std(class_f1s):.4f}")

print(f"\n  总耗时: {elapsed}")
print(f"  模型文件: fold_1.pth ~ fold_{N_FOLDS}.pth")

# ─── 写入 kfold_log.csv ───
summary_row = {
    "时间": start_time.strftime("%Y-%m-%d %H:%M:%S"),
    "折数": N_FOLDS,
    "全量准确率均值": f"{np.mean(full_accs):.4f}",
    "全量准确率标准差": f"{np.std(full_accs):.4f}",
    "留出验证均值": f"{np.mean(val_accs):.4f}",
    "留出验证标准差": f"{np.std(val_accs):.4f}",
    "留出验证宏平均F1均值": f"{np.mean(macro_f1s):.4f}",
}

for r in results:
    summary_row[f"fold{r['fold']}_full_acc"] = f"{r['full_accuracy']:.4f}"
    summary_row[f"fold{r['fold']}_val_acc"] = f"{r['val_accuracy']:.4f}"

write_header = not KFOLD_LOG.exists()
try:
    with open(KFOLD_LOG, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(summary_row)
    print(f"\n📊 交叉验证日志: {KFOLD_LOG.name}")
except PermissionError:
    alt = BASE / f"kfold_log_{start_time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(alt, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        w.writeheader()
        w.writerow(summary_row)
    print(f"⚠️  {KFOLD_LOG.name} 被占用，已保存至: {alt.name}")
