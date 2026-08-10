#!/usr/bin/env bash
# 🎃 抽卡循环：连续跑 N 次 train_3class.py，自动记录每次结果
# 用法:  bash roll_loop.sh [次数]     （默认 5 次）
# 说明:  每次约 20 分钟，跑完自动更新 Top-3 归档并写 training_log.csv
#        中途某次失败会跳过继续，不会中断整轮
cd "$(dirname "$0")"
N="${1:-5}"
PY="C:/Users/LTY/anaconda3/python.exe"

echo "🎃 开始抽卡，共 $N 次（每次约 20 分钟，此窗口请勿关闭）"
for ((i=1; i<=N; i++)); do
    echo ""
    echo "================================================"
    echo "  第 $i / $N 次   开始于 $(date '+%H:%M:%S')"
    echo "================================================"
    "$PY" train_3class.py
    echo "    ✅ 第 $i 次结束于 $(date '+%H:%M:%S')"
done

echo ""
echo "🎉 全部 $N 次抽卡完成！最后 $N 条训练记录："
tail -n "$N" training_log.csv
