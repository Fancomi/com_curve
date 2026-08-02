"""蹦床线的图表。只吃 CSV + JSON，不需要 FBX SDK。

三张图，各承载一件事：

  curves.png     四条曲线差多少：高度叠放 / 单段放大 / 两个误差源 / 段质量占比
  trajectory.png 轨迹长什么样：X-Y 按时间上色 / 各段腾空的高度-时间曲线叠放
  retarget.png   反算改了什么：被盯曲线 / 骨盆 / Δroot / 垂直速度

放大图只画一段腾空而不是全片：全片 20 s 里几条曲线在 ±25 cm 内高频缠绕，除了
"它们很像"读不出别的，而那个结论已经由高度叠放图承担了。
"""
import numpy as np

from ..common import plotstyle as PS
from ..common.plotstyle import plt
from . import curves as Cv
from . import stage as St

LABEL = {
    "root": "根骨骼 Root_M（骨盆）",
    "mesh": "Mesh 体积重心（每帧积分）",
    "mujoco": "仿 MuJoCo（14 刚体 · 网格质量）",
    "anthro": "真实人体（Dempster 体段质量）",
}


def _longest(meta):
    """腾空最长的那一段：肢体动作最充分，各口径的分歧最大。"""
    return max(meta["gravity"]["flights"], key=lambda pair: pair[1] - pair[0])


def _window(data, meta, pad=4):
    start, stop = _longest(meta)
    frames = data["frame"]
    a = max(int(np.searchsorted(frames, start)) - pad, 0)
    b = int(np.searchsorted(frames, stop)) + pad + 1
    return a, b, start, stop


def _bands(axis, data, meta, label=None):
    PS.shade_spans(axis, data["time_s"], data["frame"], meta["contacts"], label)


def height_axis(axis, data, meta, keys, cjk):
    _bands(axis, data, meta, "触床（床面被压）" if cjk else "contact")
    for key in keys:
        axis.plot(data["time_s"], data[f"{key}_y_cm"], color=PS.COLOR[key], lw=1.4,
                  label=LABEL[key] if cjk else key)
    axis.axhline(meta["bed_rest_y_cm"], color=PS.DATUM, ls="--", lw=1.0)
    axis.set_ylabel("高度 Y (cm)" if cjk else "height Y (cm)")
    axis.set_title("四种口径的重心高度（全片尺度下几乎重合）" if cjk
                   else "centre-of-mass height, four definitions")
    axis.legend(loc="lower right", fontsize=8, ncol=2)
    axis.grid(alpha=0.25)


def offset_axis(axis, data, meta, keys, cjk):
    a, b, start, stop = _window(data, meta)
    axis.axhline(0, color="#555", lw=0.8)
    for key in keys[1:]:
        axis.plot(data["time_s"][a:b],
                  data[f"{key}_y_cm"][a:b] - data["root_y_cm"][a:b],
                  color=PS.COLOR[key], lw=1.8, label=LABEL[key] if cjk else key)
    axis.set_xlabel("时间 (s)" if cjk else "time (s)")
    axis.set_ylabel("相对 root 的高度差 (cm)" if cjk else "minus pelvis (cm)")
    axis.set_title(f"放大一段腾空（f{start}–{stop}）：三条与骨盆的差" if cjk
                   else f"one flight (f{start}-{stop}) relative to the pelvis")
    axis.legend(loc="upper right", fontsize=8)
    axis.grid(alpha=0.25)


def residual_axis(axis, data, meta, cjk):
    """两个误差源分开看：刚体近似的代价 vs 体段密度的代价。"""
    a, b, _start, _stop = _window(data, meta)
    stats = meta["pairwise"]
    axis.axhline(0, color="#555", lw=0.8)
    pairs = (("mujoco", "mesh", "mesh_to_mujoco", "刚体近似的代价"),
             ("anthro", "mujoco", "mujoco_to_anthro", "体段密度的代价"))
    for high, low, key, text in pairs:
        entry = stats[key]
        label = (f"{high} − {low}：{text}（全片 σ={entry['dy_std_cm']:.2f} cm，"
                 f"max {entry['dy_max_abs_cm']:.1f}）") if cjk else key
        axis.plot(data["time_s"][a:b],
                  data[f"{high}_y_cm"][a:b] - data[f"{low}_y_cm"][a:b],
                  color=PS.COLOR[high], lw=1.8, label=label)
    axis.set_xlabel("时间 (s)" if cjk else "time (s)")
    axis.set_ylabel("高度差 (cm)" if cjk else "difference (cm)")
    axis.set_title("同一段腾空：两个误差源分开看" if cjk
                   else "the two error sources, same flight")
    axis.legend(loc="upper right", fontsize=8)
    axis.grid(alpha=0.25)


def mass_axis(axis, facts, cjk):
    """段质量占比对照：网格体素给的 vs Dempster 表给的。

    这解释了 anthro 为什么与 mujoco 分歧——躯干占比 38% vs 50%，四肢相应偏重。
    """
    entries = facts["segments"]
    positions = np.arange(len(entries))
    axis.barh(positions - 0.2, [e["mesh_mass_pct"] for e in entries], height=0.38,
              color=PS.COLOR["mujoco"], label="网格体素（均质）" if cjk else "voxels")
    axis.barh(positions + 0.2, [e["dempster_mass_pct"] for e in entries],
              height=0.38, color=PS.COLOR["anthro"],
              label="Dempster 表" if cjk else "Dempster")
    axis.set_yticks(positions)
    axis.set_yticklabels([e["name"] for e in entries], fontsize=8)
    axis.invert_yaxis()
    axis.set_xlabel("占全身质量 (%)" if cjk else "share of body mass (%)")
    axis.set_title(f"段质量占比（全身 {facts['total_mass_kg']:.1f} kg "
                   f"@ ρ={facts['density_kg_m3']:g}）" if cjk
                   else "segment mass share")
    axis.legend(loc="lower right", fontsize=8)
    axis.grid(alpha=0.25, axis="x")


def curves_figure(data, meta, path, cjk):
    keys = tuple(meta["modes"])
    figure, axes = plt.subplots(2, 2, figsize=(16, 10))
    height_axis(axes[0][0], data, meta, keys, cjk)
    offset_axis(axes[0][1], data, meta, keys, cjk)
    residual_axis(axes[1][0], data, meta, cjk)
    mass_axis(axes[1][1], meta["scene"]["rigid_bodies"], cjk)
    gravity = meta["gravity"]["per_curve"]
    summary = "   ".join(f"{k} {gravity[k]['gravity_mean']:+.2f}" for k in keys)
    return PS.finish(figure, path,
                     (f"蹦床重心：四种计算口径 · {meta['frames'][1]} 帧 @ "
                      f"{meta['fps']:g} fps · 腾空段重力拟合 {summary} m/s²") if cjk
                     else f"four definitions · gravity fit {summary}")


def trajectory_figure(data, meta, path, cjk):
    """左：X-Y 轨迹按时间上色。右：各段腾空的高度-时间曲线叠放。

    右图故意不画 X-Y：十次弹跳都在同一点附近起跳，弧线在 X-Y 上必然互相穿插成一团
    （试过，读不出任何东西）。改成以离床时刻为零点的高度-时间，抛物线族立刻对齐。
    """
    from matplotlib.collections import LineCollection

    figure, axes = plt.subplots(1, 2, figsize=(15, 6.5),
                                gridspec_kw={"width_ratios": [1, 1.1]})
    left = axes[0]
    points = np.column_stack([data["mesh_x_cm"], data["mesh_y_cm"]])
    collection = LineCollection(np.stack([points[:-1], points[1:]], axis=1),
                                cmap="viridis", linewidths=1.6)
    collection.set_array(data["time_s"][:-1])
    left.add_collection(collection)
    left.plot(data["root_x_cm"], data["root_y_cm"], color="#8a8a8a", lw=0.7,
              alpha=0.75, zorder=1, label=LABEL["root"] if cjk else "root")
    touching = data["contact_weight"] > 1e-6
    left.scatter(data["mesh_x_cm"][touching], data["mesh_y_cm"][touching], s=12,
                 facecolor="none", edgecolor="#b3272d", lw=0.9, zorder=4,
                 label="触床帧" if cjk else "contact frames")
    left.axhline(meta["bed_rest_y_cm"], color=PS.DATUM, ls="--", lw=1.0)
    figure.colorbar(collection, ax=left, pad=0.02).set_label(
        "时间 (s)" if cjk else "time (s)")
    left.set_xlabel("X (cm)")
    left.set_ylabel("Y (cm)")
    left.set_title("Mesh 重心轨迹（颜色＝时间）" if cjk else "mesh COM path")
    left.autoscale_view()
    left.set_aspect("equal", adjustable="datalim")
    left.legend(loc="lower left", fontsize=9)
    left.grid(alpha=0.25)

    right = axes[1]
    frames, dt = data["frame"], 1.0 / meta["fps"]
    trim = meta["gravity"]["trim_frames"]
    flights = meta["gravity"]["flights"]
    colours = plt.get_cmap("viridis")(np.linspace(0, 0.92, len(flights)))
    for index, (colour, (start, stop)) in enumerate(zip(colours, flights)):
        a = int(np.searchsorted(frames, start))
        b = int(np.searchsorted(frames, stop)) + 1
        height = data["mesh_y_cm"][a:b]
        clock = np.arange(len(height)) * dt
        # 拟合用的中段先垫一层半透明粗线：那个 −8.8 m/s² 是从哪几帧算出来的要看得见。
        right.plot(clock[trim:len(height) - trim],
                   height[trim:len(height) - trim], color=colour, lw=5.0,
                   alpha=0.28, solid_capstyle="round", zorder=1)
        right.plot(clock, height, color=colour, lw=1.4, alpha=0.95, zorder=2,
                   label=f"#{index + 1}  f{start}–{stop}")
    right.axhline(meta["bed_rest_y_cm"], color=PS.DATUM, ls="--", lw=1.0)
    right.set_xlabel("离床后时间 (s)" if cjk else "time since take-off (s)")
    right.set_ylabel("Mesh 重心高度 (cm)" if cjk else "mesh COM height (cm)")
    fit = meta["gravity"]["per_curve"]["mesh"]["gravity_mean"]
    right.set_title(f"{len(flights)} 段腾空（粗线＝拟合区间，g={fit:+.2f} m/s²）"
                    if cjk else f"{len(flights)} flight arcs")
    right.legend(loc="lower center", ncol=4, fontsize=7.5, framealpha=0.9)
    right.grid(alpha=0.25)
    return PS.finish(figure, path)


def retarget_figure(data, meta, curve_meta, path, cjk):
    target = meta["target"]
    figure, axes = plt.subplots(2, 2, figsize=(16, 9.5), sharex=True)

    for axis, (before, after, title) in zip(
            (axes[0][0], axes[0][1]),
            ((f"{target}_y_cm", f"fixed_{target}_y_cm",
              f"被盯曲线（{target}）：修正后是严格抛物线" if cjk else "tracked curve"),
             ("root_y_cm", "fixed_root_y_cm",
              "骨盆高度：实际被改的就是这条曲线" if cjk else "pelvis height"))):
        PS.shade_spans(axis, data["time_s"], data["frame"], curve_meta["contacts"],
                       "触床" if cjk else "contact")
        axis.plot(data["time_s"], data[before], color=PS.BEFORE, lw=1.5,
                  label="原始（骨盆按抛物线 K）" if cjk else "original")
        axis.plot(data["time_s"], data[after], color=PS.COLOR[target], lw=1.5,
                  label="修正后（被盯曲线按抛物线）" if cjk else "corrected")
        axis.axhline(curve_meta["bed_rest_y_cm"], color=PS.DATUM, ls="--", lw=1.0)
        axis.set_ylabel("高度 Y (cm)" if cjk else "height Y (cm)")
        axis.set_title(title)
        axis.legend(loc="lower right", fontsize=8)
        axis.grid(alpha=0.25)

    shift = axes[1][0]
    PS.shade_spans(shift, data["time_s"], data["frame"], curve_meta["contacts"])
    shift.axhline(0, color="#555", lw=0.8)
    for column, colour, label in (("shift_y_cm", "#b3272d", "ΔY"),
                                  ("shift_x_cm", "#3a3a3a", "ΔX"),
                                  ("shift_z_cm", "#2f6f8f", "ΔZ")):
        shift.plot(data["time_s"], data[column], color=colour, lw=1.2, label=label)
    shift.set_xlabel("时间 (s)" if cjk else "time (s)")
    shift.set_ylabel("根骨骼修正量 (cm)" if cjk else "root correction (cm)")
    shift.set_title(f"Δroot：触床帧恒为 0，最大 {meta['shift_max_cm']:.0f} cm" if cjk
                    else "root correction")
    shift.legend(loc="upper right", fontsize=8, ncol=3)
    shift.grid(alpha=0.25)

    speed = axes[1][1]
    PS.shade_spans(speed, data["time_s"], data["frame"], curve_meta["contacts"])
    speed.axhline(0, color="#555", lw=0.8)
    speed.plot(data["time_s"], data["vy_mps"], color=PS.BEFORE, lw=1.4,
               label="原始" if cjk else "original")
    speed.plot(data["time_s"], data["fixed_vy_mps"], color=PS.COLOR[target], lw=1.4,
               label="修正后" if cjk else "corrected")
    speed.set_xlabel("时间 (s)" if cjk else "time (s)")
    speed.set_ylabel("被盯曲线垂直速度 (m/s)" if cjk else "vertical speed (m/s)")
    # 只统计真正的起跳；首段是从空中开始的，没有起跳。
    gains = [f["takeoff_after_mps"] - f["takeoff_before_mps"]
             for f in meta["flights"] if f["takeoff_after_mps"] > 0]
    speed.set_title(f"修正后每段是斜率 -9.81 的直线；离床速度平均 "
                    f"+{np.mean(gains):.1f} m/s" if cjk else "vertical speed")
    speed.legend(loc="upper right", fontsize=8)
    speed.grid(alpha=0.25)

    apex_before = max(f["apex_before_m"] for f in meta["flights"])
    apex_after = max(f["apex_after_m"] for f in meta["flights"])
    return PS.finish(figure, path,
                     (f"反算根骨骼 · 目标 {target} · g = -{meta['gravity_cm_s2'] / 100:.2f}"
                      f" m/s² · 最高点 {apex_before:.2f} -> {apex_after:.2f} m · "
                      f"修正后实测 {meta['gravity_after_mps2']:+.4f}"
                      f"（残差 {meta['gravity_after_rms_cm']:.0e} cm）") if cjk
                     else f"root retargeted so {target} follows a true parabola",
                     rect=(0, 0, 1, 0.96))
