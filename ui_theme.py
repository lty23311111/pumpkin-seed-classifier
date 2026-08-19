"""公共 UI 主题定义。暗色调 Tkinter 界面统一风格。

集中管理颜色常量、字体名、通用辅助函数，
供 gui.py / demo.py / annotate_3class.py / review_3class.py 共用。
"""

# ═══════════════════════════════════════════════════════════════
#  颜色
# ═══════════════════════════════════════════════════════════════

# 完整色板（gui.py 用字典风格）
C = {
    "bg":           "#1f1f1f",
    "surface":      "#2c2c2c",
    "surface2":     "#363636",
    "border":       "#3e3e3e",
    "text":         "#f0f0f0",
    "text2":        "#b0b0b0",
    "text3":        "#7a7a7a",
    "accent":       "#ff8c00",
    "accent_hover": "#ffa726",
    "green":        "#4caf50",
    "amber":        "#ff9800",
    "orange":       "#ff7043",
    "red":          "#ef5350",
    "green_bg":     "#1e3520",
    "amber_bg":     "#3d301e",
    "orange_bg":    "#3d241e",
    "red_bg":       "#3d1e1e",
}

# 便捷别名（demo.py / annotate_3class.py 用独立变量风格）
C_BG   = C["bg"]
C_SFC  = C["surface"]
C_TXT  = C["text"]
C_TXT2 = C["text2"]
C_TXT3 = C["text3"]

# 等级颜色
GRADE_COLORS     = {0: "#4caf50", 1: "#ff9800", 2: "#ef5350"}
GRADE_COLORS_1   = {1: "#4caf50", 2: "#ff9800", 3: "#ef5350"}   # 1-indexed
GRADE_BG         = {0: "#1e3520", 1: "#3d301e", 2: "#3d1e1e"}
GRADE_BG_LIST    = ["#1e3520", "#3d301e", "#3d1e1e"]

CLASS_COLORS     = ["#4caf50", "#ff9800", "#ef5350"]             # 列表形式

# ═══════════════════════════════════════════════════════════════
#  字体
# ═══════════════════════════════════════════════════════════════

FONT = "Microsoft YaHei UI"
MONO = "Cascadia Code"

# ═══════════════════════════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════════════════════════


def lighter(hex_color: str, amount: int = 38) -> str:
    """将十六进制颜色变亮。amount 为 RGB 各通道增加的值。"""
    r = min(255, int(hex_color[1:3], 16) + amount)
    g = min(255, int(hex_color[3:5], 16) + amount)
    b = min(255, int(hex_color[5:7], 16) + amount)
    return f"#{r:02x}{g:02x}{b:02x}"
