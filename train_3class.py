"""南瓜子外观质量分类 — 3分类。ResNet-50 + 均衡采样。

配方与西瓜子项目一致：ResNet-50 + WeightedRandomSampler + 普通交叉熵
+ CosineAnnealingLR，自动维护 Top-3 最佳模型并记录训练日志。
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
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

BASE = Path(__file__).parent
# 使用 512/1024 预处理缓存加速（verify_cache.py 验证 1024 与全分辨率一致率 99.83%）
# 若要改回全分辨率原始图，把 ANNOTATED 改回 "annotated_3class" 即可
ANNOTATED = BASE / "data" / "cache_1024"
MODEL_DIR = BASE / "models"
LOG_PATH  = BASE / "training_log.csv"
MODEL_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 60
LR = 0.001
IMG_SIZE = 224
NUM_CLASSES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["一级·完好", "二级·轻微瑕疵", "三级·明显瑕疵"]

# ─── 数据增强（与西瓜子项目一致） ───
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

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))
train_idx = train_idx.tolist()
val_idx = val_idx.tolist()

train_ds = Subset(datasets.ImageFolder(root=str(ANNOTATED), transform=train_transform), train_idx)
val_ds   = Subset(datasets.ImageFolder(root=str(ANNOTATED), transform=val_transform), val_idx)

# 均衡采样
train_targets = targets[train_idx]
class_sample_counts = [int((train_targets == i).sum()) for i in range(NUM_CLASSES)]
sample_weights = [1.0 / class_sample_counts[t] for t in train_targets]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_targets), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"  训练集: {len(train_ds)}  验证集: {len(val_ds)}")

# ─── ResNet-50 ───
print(f"\n构建模型 (设备: {DEVICE})...")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
num_features = model.fc.in_features
model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(num_features, NUM_CLASSES))
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ─── 训练 ───
print(f"\n开始训练 ({EPOCHS} epochs)...")
print("-" * 60)

best_acc = 0.0
MODEL_PATHS = [
    MODEL_DIR / "best_model_3class.pth",      # 第 1 名
    MODEL_DIR / "best_model_3class_2.pth",    # 第 2 名
    MODEL_DIR / "best_model_3class_3.pth",    # 第 3 名
]
record_path = MODEL_DIR / "_best_record_3class.json"
temp_path   = MODEL_DIR / "_temp_3class.pth"

pbar = tqdm(range(EPOCHS), desc="训练", ncols=90)

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

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), temp_path)

    pbar.set_postfix(Train=f"{train_acc:.3f}", Val=f"{val_acc:.3f}", Best=f"{best_acc:.3f}")

# ─── 结果 ───
print(f"\n========== 训练完成 ==========")
print(f"最佳验证准确率: {best_acc:.4f}")

model.load_state_dict(torch.load(temp_path, map_location=DEVICE))
model.eval()
temp_path.unlink(missing_ok=True)  # 清理临时最佳权重

all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        all_preds.extend(outputs.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

print("\n分类报告 (验证集):")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=3))
print("混淆矩阵:")
cm = confusion_matrix(all_labels, all_preds)
for i, name in enumerate(CLASS_NAMES):
    print(f"  实际{name[:4]}  " + "  ".join(f"{cm[i][j]:4d}" for j in range(NUM_CLASSES)))

# ─── 全量评估 ───
print(f"\n全量评估 (全部 {len(full_dataset)} 张)...")
full_ds = datasets.ImageFolder(root=str(ANNOTATED), transform=val_transform)
full_loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=False)
all_preds_f, all_labels_f = [], []
with torch.no_grad():
    for inputs, labels in full_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        all_preds_f.extend(outputs.argmax(1).cpu().tolist())
        all_labels_f.extend(labels.cpu().tolist())

new_acc = (np.array(all_labels_f) == np.array(all_preds_f)).mean()
print(classification_report(all_labels_f, all_preds_f, target_names=CLASS_NAMES, digits=3))

# ── 归档（Top-3 排名） ──
print(f"\n========== 模型归档 ==========")
print(f"本次全量准确率（仅用于检查拟合，不用于模型排名）: {new_acc:.4f}")

# 读取已有记录（兼容旧格式）
if record_path.exists():
    raw = json.loads(record_path.read_text())
    if isinstance(raw, dict) and raw.get("metric") == "validation_accuracy":
        records = raw.get("records", [])
    else:
        records = []
else:
    records = []

old_count = len(records)

# 与已有记录比较，插入正确位置（降序）
inserted = False
for i, r in enumerate(records):
    if best_acc > r["accuracy"]:
        records.insert(i, {"accuracy": best_acc})
        inserted = True
        rank = i + 1
        break
if not inserted:
    records.append({"accuracy": best_acc})
    rank = len(records)

records = records[:3]
record_path.write_text(json.dumps({"metric": "validation_accuracy", "records": records},
                                  indent=2, ensure_ascii=False), encoding="utf-8")

# ── 模型文件移位：排名与文件始终对齐 ──
written = set()  # 记录本次移位写入的位置，cleanup 跳过它们

if rank == 1:
    if old_count >= 3 and MODEL_PATHS[2].exists():
        MODEL_PATHS[2].unlink()
    if old_count >= 2 and MODEL_PATHS[1].exists():
        MODEL_PATHS[1].rename(MODEL_PATHS[2])
        written.add(2)
    if MODEL_PATHS[0].exists():
        MODEL_PATHS[0].rename(MODEL_PATHS[1])
        written.add(1)
    torch.save(model.state_dict(), MODEL_PATHS[0])
    written.add(0)
    print(f"  🏆 第 1 名: {MODEL_PATHS[0].name}  ({best_acc:.4f}) ← 本次训练")
elif rank == 2:
    if old_count >= 3 and MODEL_PATHS[2].exists():
        MODEL_PATHS[2].unlink()
    if old_count >= 2 and MODEL_PATHS[1].exists():
        MODEL_PATHS[1].rename(MODEL_PATHS[2])
        written.add(2)
    torch.save(model.state_dict(), MODEL_PATHS[1])
    written.add(1)
    print(f"  🏆 第 2 名: {MODEL_PATHS[1].name}  ({best_acc:.4f}) ← 本次训练")
elif rank == 3:
    torch.save(model.state_dict(), MODEL_PATHS[2])
    written.add(2)
    print(f"  🏆 第 3 名: {MODEL_PATHS[2].name}  ({best_acc:.4f}) ← 本次训练")
else:
    print(f"  ⏭️  未进 Top-3（排名 {rank}），跳过保存")

# 打印完整排行榜
for i, r in enumerate(records):
    model_path = MODEL_PATHS[i]
    acc = r["accuracy"]
    if i + 1 != rank:
        status = f"({acc:.4f})" if model_path.exists() else f"({acc:.4f}) ⚠️ 缺失"
        print(f"    第 {i+1} 名: {model_path.name}  {status}")

# 清理冗余文件
for i in range(len(records), len(MODEL_PATHS)):
    if i not in written and MODEL_PATHS[i].exists():
        MODEL_PATHS[i].unlink()
        print(f"  🗑️  淘汰: {MODEL_PATHS[i].name}")

top3_str = ", ".join(f"{r['accuracy']:.4f}" for r in records)
print(f"\n当前 Top-3 排行榜: [{top3_str}]")

# ─── 训练日志 ───
report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4,
                               output_dict=True, zero_division=0)

row = {
    "时间": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "全量准确率（拟合检查）": f"{new_acc:.4f}",
    "验证准确率": f"{best_acc:.4f}",
    "宏平均F1": f"{report['macro avg']['f1-score']:.4f}",
}
for i, name in enumerate(CLASS_NAMES):
    row[f"{name}_精确率"] = f"{report[name]['precision']:.4f}"
    row[f"{name}_召回率"] = f"{report[name]['recall']:.4f}"
    row[f"{name}_F1"] = f"{report[name]['f1-score']:.4f}"
row["备注"] = f"🏆 第{rank}名" if rank <= len(MODEL_PATHS) else "跳过"

write_header = not LOG_PATH.exists()
try:
    with open(LOG_PATH, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"📊 训练日志: {LOG_PATH.name}")
except PermissionError:
    # 文件被 VS Code/Excel 锁了，存到临时文件
    alt_path = BASE / f"training_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(alt_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    print(f"⚠️  {LOG_PATH.name} 被占用，已保存至: {alt_path.name}")
    print(f"   关闭该文件后，将 {alt_path.name} 的内容追加到 {LOG_PATH.name} 即可")
