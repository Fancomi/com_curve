"""dive 线的渲染件：正交四格与叠回源视频。

正交四格与 tramp 共用 `common.grid`，这里只提供 dive 特有的东西：
视角轴向、颜色、事件角标，以及叠加视频的曲线板与 HUD。

**正视格与源视频叠加效果逐像素一致**：几何来自 `align.place_on_plane`，X/Y 是每帧顶点
在源视频里的真实投影像素反投到固定深度平面的结果，正交相机的水平/垂直轴就是像素轴的
线性映射。

**侧视格只有体厚，没有前后位移**：单目拿不到深度，`cam_t.z` 在这段片子里从 51 m 抖到
110 m，把它当轨迹用就是把噪声画成运动。侧视图看的是"团身有多紧"，不是"人往前走了多少"。
"""
import cv2
import numpy as np

from ..common import grid as Grid
from ..common import plotstyle as PS
from ..common import raster as R
from . import align as A

# (键, 水平世界轴, 深度轴, 深度朝向)。global 是 X 右 / Y 上 / Z 朝相机，
# 所以正视相机在 +Z，侧视在 +X。
VIEWS = (("front", 0, 2, +1), ("side", 2, 0, +1))
VIEW_CN = {"front": "正视", "side": "侧视"}
VIEW_SUB = {"front": "与源视频投影一致", "side": "仅体厚 · 单目无深度"}
MODE_CN = {"mesh": "Mesh 重心", "gb": "国标 15 段"}
MODE_SUB = {"mesh": "均质体积重心", "gb": "GB/T 17245 · 体表骨突"}

MARKER = {key: PS.bgr(key) for key in A.MODES}
# 正交渲染用淡蓝白；叠在源视频上换紫罗兰——这个场馆墙是米白、池是青，淡蓝两边都撞，
# 紫罗兰在两者之间没有邻居，半透明叠上去人体轮廓仍清楚。
BODY_TINT = np.array([176, 190, 226], dtype=np.float64)
OVERLAY_TINT = np.array([214, 132, 196], dtype=np.float64)
SILHOUETTE = (245, 190, 235)
MESH_ALPHA = 0.62
EVENT_COLOUR = {"board": (150, 150, 150), "jump": (255, 160, 60),
                "peak": (40, 170, 255), "water": (60, 60, 230)}
EVENT_CN = {"board": "起跳板", "jump": "离板", "peak": "最高点", "water": "入水"}
EVENT_KEYS = (("board", "board_frame"), ("jump", "jumping_frame"),
              ("peak", "peak_frame"), ("water", "water_entry_frame"))
MIN_WIDTH = 300
TRAIL_SPAN = 90              # 1.5 s @ 60 fps

PANEL_BG = (30, 27, 24)
PANEL_GRID = (58, 54, 50)
TEXT = (230, 228, 224)
MUTED = (156, 152, 148)


def views():
    return [(key, VIEW_CN[key]) for key, *_rest in VIEWS]


def cameras(clip, height):
    """两台正交相机，共用垂直范围。"""
    clouds = [m for m in clip.meshes if m is not None]
    tracks = [getattr(clip, key) for key in clip.modes]
    ranges, y_range = Grid.frame_ranges(clouds, tracks, pad=0.10)
    return {key: R.OrthoCamera(axis, depth_axis, sign, ranges[axis], y_range,
                               height, MIN_WIDTH)
            for key, axis, depth_axis, sign in VIEWS}


def event_badge(width, index, events):
    """横幅右侧的事件角标。命中窗口 ±1 帧。"""
    for name, key in EVENT_KEYS:
        frame = events.get(key)
        if frame is not None and abs(index - int(frame)) <= 1:
            return [(width - 150, 9, EVENT_CN[name], EVENT_COLOUR[name], "small")]
    return []


def draw_panel(panel, curves, meta, index):
    """右侧曲线板：上高度、下垂直速度，当前帧游标。"""
    panel[:] = PANEL_BG
    height, width = panel.shape[:2]
    fps = float(meta["fps"])
    frames = curves["frames"]
    time = frames / fps
    span = max(float(time[-1]), 1e-6)
    events = meta.get("events") or {}
    flight = (meta.get("flight") or {}).get("flight") or []
    left, right = 52, width - 18

    def x_of(clock):
        return int(left + (right - left) * clock / span)

    def block(top, bottom, series_map, label):
        pool = np.concatenate([s[np.isfinite(s)] for s in series_map.values()])
        low, high = ((float(pool.min()), float(pool.max())) if len(pool)
                     else (-1.0, 1.0))
        pad = max(0.12 * (high - low), 0.15)
        low, high = low - pad, high + pad

        def y_of(value):
            return int(bottom - (value - low) / (high - low) * (bottom - top))

        if len(flight) == 2:
            cv2.rectangle(panel, (x_of(flight[0] / fps), top),
                          (x_of(flight[1] / fps), bottom), (46, 43, 40), -1)
        for step in range(5):
            row = top + int((bottom - top) * step / 4)
            cv2.line(panel, (left, row), (right, row), PANEL_GRID, 1, cv2.LINE_AA)
            cv2.putText(panel, f"{high - (high - low) * step / 4:+.1f}",
                        (6, row + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, MUTED, 1,
                        cv2.LINE_AA)
        cv2.line(panel, (left, y_of(0.0)), (right, y_of(0.0)), (90, 130, 100), 1,
                 cv2.LINE_AA)
        for name, key in EVENT_KEYS:
            frame = events.get(key)
            if frame is None:
                continue
            column = x_of(float(frame) / fps)
            cv2.line(panel, (column, top), (column, bottom),
                     EVENT_COLOUR[name], 1, cv2.LINE_AA)
        for name, series in series_map.items():
            points = []
            for step in range(len(series)):
                if not np.isfinite(series[step]):
                    if len(points) > 1:
                        cv2.polylines(panel, [np.array(points, np.int32)], False,
                                      MARKER[name], 2, cv2.LINE_AA)
                    points = []
                    continue
                points.append((x_of(time[step]), y_of(series[step])))
            if len(points) > 1:
                cv2.polylines(panel, [np.array(points, np.int32)], False,
                              MARKER[name], 2, cv2.LINE_AA)
        cursor = x_of(index / fps)
        cv2.line(panel, (cursor, top), (cursor, bottom), (215, 213, 210), 1,
                 cv2.LINE_AA)
        for name, series in series_map.items():
            if index < len(series) and np.isfinite(series[index]):
                cv2.circle(panel, (cursor, y_of(series[index])), 5, MARKER[name],
                           -1, cv2.LINE_AA)
        cv2.putText(panel, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, TEXT, 1, cv2.LINE_AA)

    split = int(height * 0.56)
    keys = [k for k in A.MODES if f"{k}_y" in curves]
    block(38, split - 34, {k: curves[f"{k}_y"] for k in keys}, "height Y (m)")
    block(split + 22, height - 46, {k: curves[f"{k}_vy"] for k in keys},
          "vertical speed (m/s)")
    for offset, name in enumerate(keys):
        cv2.putText(panel, name, (left + offset * 66, height - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, MARKER[name], 1, cv2.LINE_AA)
    return panel


def draw_hud(frame, painter, curves, meta, index, valid):
    """左下角状态 + 事件角标。

    放左下而不是左上：原始机位视频左上角烧了一个巨大的时间戳（约 700×60 px），叠上去
    两行字互相穿过谁都读不了。底部那条池边是画面里最空的区域。加一层半透明底板，
    浅色池水上白字才有对比。
    """
    fps = float(meta["fps"])
    alignment = meta.get("alignment") or {}
    block = ((meta.get("flight") or {}).get("per_curve") or {}).get("mesh") or {}
    top = frame.shape[0] - 118
    plate = frame[top:frame.shape[0] - 10, 12:560]
    cv2.addWeighted(plate, 0.28, np.zeros_like(plate), 0.72, 0.0, dst=plate)

    lines = [(24, top + 6, "跳水质心叠加 · 单目 HSMR", TEXT, "title"),
             (24, top + 34,
              f"f{index}  t={index / fps:.2f}s   尺度 "
              f"{alignment.get('scale_mode', '?')} "
              f"{alignment.get('scale_m_per_px', 0):.5f} m/px   体高 "
              f"{alignment.get('body_height_m', 0):.2f} m",
              (198, 196, 192), "small")]
    if block:
        lines.append((24, top + 56,
                      f"腾空段 mesh 拟合 g={block.get('gravity_trimmed_mps2', 0):+.2f}"
                      f" m/s2   顶点 {block.get('apex_m', 0):.2f} m",
                      (198, 196, 192), "small"))
    if valid:
        mesh_y = curves["mesh_y"][index]
        text = f"mesh Y {mesh_y:+.2f} m"
        # gb 可能没采（无标注），也可能该帧没算出来；两种情况都只报 mesh。
        gb_y = curves["gb_y"][index] if "gb_y" in curves else np.nan
        if np.isfinite(gb_y):
            text += (f"    国标 Y {gb_y:+.2f} m    差 "
                     f"{(gb_y - mesh_y) * 100:+.1f} cm")
        lines.append((24, top + 78, text, MARKER["mesh"], "small"))
    else:
        lines.append((24, top + 78, "该帧无 mesh（入水后）", MUTED, "small"))
    for name, key in EVENT_KEYS:
        frame_no = (meta.get("events") or {}).get(key)
        if frame_no is not None and abs(index - int(frame_no)) <= 1:
            lines.append((frame.shape[1] - 240, frame.shape[0] - 60,
                          EVENT_CN[name], EVENT_COLOUR[name], "title"))
    painter.paint(frame, lines)
