# 🎃 南瓜子外观质量分类

用海康工业相机逐颗拍摄南瓜子，通过深度学习（ResNet-50）自动将南瓜子分为 **3 个外观质量等级**。基于西瓜子项目（4 分类）验证过的完整流水线复刻，并新增了预处理缓存优化。

---

## 最终成绩

| 指标 | 数值 |
|------|------|
| 数据集 | 600 张 BMP（海康相机，固定光源，3072×2048） |
| 分类 | 3 级（一级·完好 / 二级·轻微瑕疵 / 三级·明显瑕疵） |
| 类别分布 | 一级 220 · 二级 201 · 三级 179 |
| **单模型最佳（验证）** | **95.00%** |
| **8 模型软投票（验证）** | **98.33%**（部署推理口径） |
| **8 模型软投票（全量）** | **99.50%**（拟合检查） |
| **K 折均值** | **92.17% ± 1.55%**（真实泛化水平） |
| 推理 | **8 模型软投票**（最佳 + 5 折 + Top-3，概率取平均） |
| 设备 | RTX 4060 Laptop (8.6GB) + CUDA 12.6 + PyTorch 2.12.1 |

## 项目结构

```
├── annotate_3class.py     # 标注工具（按键 1/2/3 分拣，支持断点续标）
├── train_3class.py        # 日常训练，自动维护 Top-3 最佳模型
├── train_kfold_3class.py  # 5 折交叉验证，评估真实泛化水平
├── review_3class.py       # 复查模型与人工标注不一致的图片
├── sync_cache.py          # 改标后同步预处理缓存（带并发锁）
├── repair_cache.py        # 缓存损坏修复（截断 PNG 从源图重建）
├── verify_cache.py        # 预处理缓存质量验证（一致率实验）
├── roll_loop.sh           # 批量重训抽卡（bash roll_loop.sh [次数]）
├── gui.py                 # 图形界面（拖拽/批量预测/CSV 导出）
├── predict.py             # 命令行推理工具（单张/文件夹）
├── demo.py                # 现场验收演示（人机对比 + 实时一致率）
├── model_utils.py          # 公共模型、预处理与软投票逻辑
├── ui_theme.py             # 图形界面共用主题与颜色配置
├── requirements.txt        # 可复现的 Python 依赖清单
├── analysis/              # 失败分析：3 张人机分歧 + 每类典型示例
├── training_log.csv       # 训练记录
├── kfold_log.csv          # 5 折交叉验证记录
└── 项目完整总结.md          # 完整项目报告
```

## 使用方法

```bash
# 安装依赖（建议在虚拟环境中执行）
pip install -r requirements.txt

# 训练（数据源已切换到预处理缓存 data/cache_1024，约 20 分钟）
python train_3class.py

# 5 折交叉验证（约 1 小时 40 分钟）
python train_kfold_3class.py

# 图形界面推理
python gui.py

# 命令行推理（单张 / 文件夹）
python predict.py <图片路径或文件夹>

# 现场验收演示
python demo.py
```

## 技术要点

| 组件 | 方案 |
|------|------|
| 模型 | ResNet-50（ImageNet1K_V2 预训练） |
| 分类头 | Dropout(0.4) + Linear(2048, 3) |
| 增强 | RandomResizedCrop + 翻转 + 旋转30° + ColorJitter + RandomErasing |
| 平衡 | WeightedRandomSampler 均衡采样 |
| 优化 | AdamW (lr=0.001) + CosineAnnealingLR (T_max=60)，60 epochs |
| 缓存 | 原图(18MB BMP) → cache_1024 PNG(565MB)，训练提速 ~5×，一致率 99.83% 实证质量中性 |

## 关联项目

- 🍉 [西瓜子外观质量分类](https://github.com/lty23311111/watermelon-seed-classifier)（4 分类，本项目的代码模板）
