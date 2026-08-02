"""matplotlib 的中文字体与全项目统一配色。

图里的文字用中文（这些图给人看，不进代码库），但**新增的图一律用英文标签**——
项目约定，见 README。已有的中文图保留。
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 四种重心口径各一色，plot 与 render 共用同一套（render 侧按 BGR 反转）。
COLOR = {"root": "#1f6feb", "mesh": "#d1642a",
         "mujoco": "#2f8f5b", "anthro": "#8b3fa8"}
# before/after 对比：before 灰蓝（"从前的样子"，不抢眼），after 用该口径原色。
BEFORE = "#6d7f92"
CONTACT = "#9aa4b2"      # 触床灰带
DATUM = "#2f6f4f"        # 床面 / 基准线


def bgr(name):
    """`COLOR` 里的十六进制 -> cv2 的 BGR 三元组。"""
    value = COLOR[name].lstrip("#")
    r, g, b = (int(value[k:k + 2], 16) for k in (0, 2, 4))
    return (b, g, r)


def pick_font():
    """挑一个装了中文字形的字体，没有就返回 None（标签退回英文）。

    顺序按字形覆盖排，不是按好看排：Songti SC 有汉字但缺 U+2212 MINUS SIGN，
    选中它会让每个负号变成豆腐块。前面几个实测都带这个字形。
    """
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("PingFang SC", "PingFang HK", "Hiragino Sans GB",
                 "Arial Unicode MS", "STHeiti", "Heiti TC", "Songti SC"):
        if name in installed:
            return name
    return None


def setup():
    """挂好中文字体，返回是否可用。"""
    font = pick_font()
    if font:
        plt.rcParams["font.family"] = font
    plt.rcParams["axes.unicode_minus"] = False
    if not font:
        print("  （没找到中文字体，标签退回英文）")
    return bool(font)


def shade_spans(axis, time, frames, spans, label=None):
    """按帧号给出的区间画成灰带（触床 / 入水）。

    第一段单独给图例，其余复用同一个颜色不再重复标注。
    """
    import numpy as np

    for index, (start, stop) in enumerate(spans):
        a = float(time[np.searchsorted(frames, start)])
        b = float(time[min(np.searchsorted(frames, stop), len(time) - 1)])
        axis.axvspan(a, b, color=CONTACT, alpha=0.22, lw=0,
                     label=label if index == 0 else None)


def finish(figure, path, title=None, rect=(0, 0, 1, 0.97)):
    """加总标题、收紧布局、存盘、关掉。返回路径。"""
    if title:
        figure.suptitle(title, fontsize=13)
        figure.tight_layout(rect=rect)
    else:
        figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
