"""dive 线的图表。只吃 CSV + JSON，不需要 frames.bin。

三张图：
  height.png      高度 / gb−mesh 差（有 gb 时）/ 垂直速度
  trajectory.png  X-Y 轨迹（按时间上色）+ 腾空段与理想抛物线对照
  scale.png       三种尺度口径下同一条曲线的样子

单位是米，Y 向上，零点是画面中心所在的水平线（不是水面）。
"""
import numpy as np

from ..common import plotstyle as PS
from ..common import signal as Sig
from ..common.plotstyle import plt
from . import align as A

LABEL = {"mesh": "Mesh 体积重心（均质）",
         "gb": "国标 GB/T 17245（15 段 · 体表骨突）"}
EVENT_STYLE = {
    "board_frame": ("起跳板", "#6b7280"),
    "jumping_frame": ("离板", "#2563eb"),
    "peak_frame": ("最高点", "#d97706"),
    "water_entry_frame": ("入水", "#dc2626"),
}
SCALE_COLOUR = {"hsmr": "#7c3aed", "metrics": "#d1642a", "gravity": "#2f6f4f"}


def curve_keys(meta):
    """这份 JSON 里实际有哪几条曲线。没给标注时 `gb` 不存在，图不该硬画。"""
    return tuple(key for key in A.MODES if key in (meta.get("modes") or {}))


def masked(data, column):
    """无效帧置 nan，matplotlib 会断线而不是连一条假的。"""
    series = data[column].copy()
    series[data["valid"] < 0.5] = np.nan
    return series


def shade_flight(axis, meta, cjk, label=True):
    """腾空弧的灰带。窗口已由 `align` 收窄到含最高点的那一条。"""
    flight = (meta.get("flight") or {}).get("flight") or []
    if len(flight) != 2:
        return
    fps = float(meta["fps"])
    axis.axvspan(flight[0] / fps, flight[1] / fps, color=PS.CONTACT, alpha=0.18,
                 lw=0, label=("腾空弧" if cjk else "flight arc") if label else None)


def mark_events(axis, meta, cjk, text=True):
    fps = float(meta["fps"])
    for key, (name, colour) in EVENT_STYLE.items():
        frame = (meta.get("events") or {}).get(key)
        if frame is None:
            continue
        clock = float(frame) / fps
        axis.axvline(clock, color=colour, ls="--", lw=1.0, alpha=0.8)
        if text:
            axis.text(clock, axis.get_ylim()[1], f" {name if cjk else key[:4]}",
                      color=colour, fontsize=8, va="top", ha="left", rotation=90)


def height_figure(data, meta, path, cjk):
    time = data["time_s"]
    modes = curve_keys(meta)
    # 差值格只在有 gb 时画：mesh 是唯一的另一条，没有可比的对象就不留空格。
    paired = "gb" in modes
    ratios = [3, 1.4, 1.6] if paired else [3, 1.6]
    figure, axes = plt.subplots(len(ratios), 1, figsize=(13, 10 if paired else 8),
                                sharex=True, gridspec_kw={"height_ratios": ratios})
    top = axes[0]
    shade_flight(top, meta, cjk)
    for key in modes:
        top.plot(time, masked(data, f"{key}_y_m"), color=PS.COLOR[key], lw=1.7,
                 label=LABEL[key] if cjk else key)
    top.axhline(0, color=PS.DATUM, ls="--", lw=1.0,
                label="画面中心高度（零点）" if cjk else "frame centre (zero)")
    alignment = meta["alignment"]
    block = (meta["flight"].get("per_curve") or {}).get("mesh") or {}
    top.set_title(
        (f"跳水重心高度 · 单目 HSMR · {meta['usable_frames']}/"
         f"{meta['divf_header']['n']} 帧有 mesh @ {meta['fps']:g} fps\n"
         f"尺度 {alignment['scale_mode']} = {alignment['scale_m_per_px']:.5f} m/px"
         f"（体高 {alignment['body_height_m']:.2f} m）· 腾空段 mesh 拟合 "
         f"{block.get('gravity_trimmed_mps2', float('nan')):+.2f} m/s²") if cjk
        else "diving centre-of-mass height")
    top.set_ylabel("高度 Y (m)" if cjk else "height Y (m)")
    top.legend(loc="lower left", fontsize=9)
    top.grid(alpha=0.25)
    mark_events(top, meta, cjk)

    if paired:
        middle = axes[1]
        shade_flight(middle, meta, cjk, label=False)
        middle.axhline(0, color="#555", lw=0.8)
        # Z 不画：固定深度平面下它只承载体厚，gb 与 mesh 的 Z 差没有物理含义。
        # 方向与 dive_summary 的 mesh_to_gb 一致（gb − mesh），两处数字能直接对上。
        for name, axis, colour in (("Y", "y", "#b3272d"), ("X", "x", "#4a4a4a")):
            middle.plot(time, masked(data, f"gb_{axis}_m")
                        - masked(data, f"mesh_{axis}_m"),
                        color=colour, lw=1.2, label=f"Δ{name}")
        middle.set_ylabel("gb - mesh (m)")
        middle.legend(loc="upper right", ncol=2, fontsize=9)
        middle.grid(alpha=0.25)

    bottom = axes[-1]
    shade_flight(bottom, meta, cjk, label=False)
    bottom.axhline(0, color="#555", lw=0.8)
    for key in modes:
        bottom.plot(time, data[f"{key}_vy_mps"], color=PS.COLOR[key], lw=1.3,
                    label=LABEL[key] if cjk else key)
    bottom.set_ylabel("垂直速度 (m/s)" if cjk else "vertical speed (m/s)")
    bottom.set_xlabel("时间 (s)" if cjk else "time (s)")
    bottom.grid(alpha=0.25)
    return PS.finish(figure, path)


def trajectory_figure(data, meta, path, cjk):
    """左：X-Y 轨迹按时间上色。右：腾空段与理想抛物线的对照。"""
    from matplotlib.collections import LineCollection

    fps = float(meta["fps"])
    modes = curve_keys(meta)
    figure, axes = plt.subplots(1, 2, figsize=(15, 6.5),
                                gridspec_kw={"width_ratios": [1, 1.15]})
    left = axes[0]
    x, y = masked(data, "mesh_x_m"), masked(data, "mesh_y_m")
    good = np.isfinite(x) & np.isfinite(y)
    points = np.column_stack([x, y])
    keep = good[:-1] & good[1:]
    collection = LineCollection(np.stack([points[:-1], points[1:]], axis=1)[keep],
                               cmap="viridis", linewidths=1.8)
    collection.set_array(data["time_s"][:-1][keep])
    left.add_collection(collection)
    for key, (name, colour) in EVENT_STYLE.items():
        frame = (meta.get("events") or {}).get(key)
        if frame is None or not (0 <= int(frame) < len(x)) or not good[int(frame)]:
            continue
        left.scatter([x[int(frame)]], [y[int(frame)]], s=46, color=colour,
                     zorder=5, label=name if cjk else key[:4])
    left.axhline(0, color=PS.DATUM, ls="--", lw=1.0)
    figure.colorbar(collection, ax=left, pad=0.02).set_label(
        "时间 (s)" if cjk else "time (s)")
    left.set_xlabel("X (m)")
    left.set_ylabel("Y (m)")
    left.set_title("Mesh 重心轨迹（颜色＝时间）" if cjk else "mesh COM path")
    left.autoscale_view()
    left.set_aspect("equal", adjustable="datalim")
    # 图例里只剩事件点；无 metrics 时一个都没有，空调 legend 会报警告。
    if left.get_legend_handles_labels()[0]:
        left.legend(loc="lower left", fontsize=8)
    left.grid(alpha=0.25)

    right = axes[1]
    analysis = meta["flight"]
    flight = analysis.get("flight") or []
    trim = int(analysis.get("trim_frames") or 8)
    if len(flight) == 2:
        first, last = int(flight[0]), int(flight[1])
        span = slice(first, last + 1)
        clock = np.arange(last - first + 1) / fps
        for key in modes:
            right.plot(clock, masked(data, f"{key}_y_m")[span], color=PS.COLOR[key],
                       lw=1.7, label=LABEL[key] if cjk else key)
        height = masked(data, "mesh_y_m")[span]
        # 拟合用的中段垫一条半透明粗线：那个 g 是从这几帧算出来的，不是整段。
        if len(height) > 2 * trim:
            right.plot(clock[trim:-trim], height[trim:-trim], color=PS.COLOR["mesh"],
                       lw=5.0, alpha=0.25, solid_capstyle="round", zorder=1,
                       label="拟合区间" if cjk else "fit window")
        # 理想抛物线：两端钉在实测位置、g = -9.81。与实测的差就是"这条曲线离纯抛体
        # 有多远"，包含尺度误差与姿态摆动两部分。
        ok = np.isfinite(height)
        if ok.sum() >= 5:
            ideal = np.full(len(height), np.nan)
            fitted = Sig.parabola_through_ends(height[ok], 1.0 / fps, A.GRAVITY)
            ideal[ok] = fitted
            right.plot(clock, ideal, color="#6b7280", ls="--", lw=1.3,
                       label="理想抛物线 g=-9.81" if cjk else "ideal parabola")
        block = (analysis.get("per_curve") or {}).get("mesh") or {}
        # 标题里说"含最高点的弧"而不是"离板→入水"：窗口是从 metrics 事件收窄来的，
        # jumping_frame 有时给的是第一次下压，那个区间里可能含好几次弹跳。
        right.set_title(
            (f"含最高点的腾空弧 f{first}–{last}（{(last - first) / fps:.2f} s）"
             f" mesh 拟合 {block.get('gravity_trimmed_mps2', float('nan')):+.2f} m/s²，"
             f"残差 {block.get('fit_rms_trimmed_m', float('nan')):.3f} m") if cjk
            else "flight arc vs ideal parabola")
    right.set_xlabel("离板后时间 (s)" if cjk else "time since take-off (s)")
    right.set_ylabel("高度 Y (m)" if cjk else "height Y (m)")
    right.legend(loc="lower center", fontsize=8)
    right.grid(alpha=0.25)
    return PS.finish(figure, path)


def scale_figure(data, meta, path, cjk):
    """三种尺度口径下的同一条曲线。

    一个标量 s 同时决定体高和轨迹幅度，两者不可能都对——这张图把这件事画出来：
    hsmr 的体高最接近成年人，但腾空段 g 偏到 -12.7；metrics 的 g 接近 -9.81，但体高
    只有 1.31 m。差的那 25% 是 HSMR 绝对深度与图像标定的不一致。
    """
    alignment = meta["alignment"]
    current = alignment["scale_mode"]
    base = alignment["scale_m_per_px"]
    body = alignment["body_height_m"]
    figure, axis = plt.subplots(figsize=(13, 6))
    shade_flight(axis, meta, cjk)
    height = masked(data, "mesh_y_m")
    for name, value in sorted((alignment.get("scale_candidates") or {}).items()):
        if value is None:
            continue
        # 高度与 s 成正比（X/Y 都是像素乘 s），换尺度只是整条曲线等比缩放。
        ratio = value / base
        axis.plot(data["time_s"], height * ratio, color=SCALE_COLOUR.get(name, "#888"),
                  lw=2.0 if name == current else 1.3,
                  ls="-" if name == current else "--",
                  label=f"{name}{'（当前）' if name == current and cjk else ''}"
                        f"  s={value:.5f} m/px  体高 {body * ratio:.2f} m")
    axis.axhline(0, color="#555", lw=0.8)
    axis.set_xlabel("时间 (s)" if cjk else "time (s)")
    axis.set_ylabel("Mesh 重心高度 Y (m)" if cjk else "mesh COM height (m)")
    axis.set_title("三种尺度口径下的同一条曲线：一个 s 同时定体高与轨迹幅度，"
                   "两者不能都对" if cjk else "the same curve under three scales")
    axis.legend(loc="upper left", fontsize=9)
    axis.grid(alpha=0.25)
    mark_events(axis, meta, cjk, text=False)
    return PS.finish(figure, path)
