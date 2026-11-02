"""dive 线的曲线导出：CSV / JSON。

单位是**米**，Y 向上，零点是源视频画面中心所在的水平线——**不是水面**，单目标定给不出
水面在哪。所以这条曲线读的是相对高度差。

速度用 `masked_derivative`：直接把无效帧填 0 再求导会在有效/无效边界造出假尖峰。
旧版 CSV 里 f437（最后一个有效帧，正是入水）的 mesh 垂直速度是 +87.8 m/s，全片其余帧
都在 20 m/s 以内——那个数是边界效应，不是运动。
"""
import numpy as np

from ..common import signal as Sig
from ..common import table as T
from . import align as A


def metrics_reference(metrics, count):
    """metrics.json 的 2D centroid 像素与"以站立点为零"的高度，作独立对照。"""
    height = np.full(count, np.nan)
    pixels = np.full((count, 2), np.nan)
    stand = (metrics.get("summary") or {}).get("standing_pos_px")
    ratio = float(metrics.get("pixel_to_meter_ratio") or 0.0)
    for index, frame in enumerate(metrics.get("frames") or []):
        if index >= count or not frame.get("has_pose"):
            continue
        centre = frame.get("centroid_2d") or [0.0, 0.0]
        if not centre[0] and not centre[1]:
            continue
        pixels[index] = centre
        if stand and ratio > 0:
            height[index] = (float(stand[1]) - float(centre[1])) * ratio
    return height, pixels


def speeds(clip):
    """{口径: (N, 3) 速度}。无效帧与段边界为 nan。"""
    valid = clip.valid == 1
    return {key: Sig.masked_derivative(getattr(clip, key), valid, 1.0 / clip.fps)
            for key in clip.modes}


def write_csv(path, clip, reference):
    """逐帧一行。列名与 tramp 那条线同构，只是单位从 cm 换成 m。"""
    height, pixels = reference
    velocity = speeds(clip)
    columns = [("frame", clip.frames),
               ("time_s", clip.frames / clip.fps),
               ("valid", clip.valid),
               ("status", clip.status)]
    for key in clip.modes:
        columns += T.vector_columns(key, getattr(clip, key), "m")
    for key in clip.modes:
        columns += [(f"{key}_vy_mps", velocity[key][:, 1]),
                    (f"{key}_speed_mps", np.linalg.norm(velocity[key], axis=1))]
    for key in clip.modes:
        columns += [(f"{key}_u_px", clip.pixels[key][:, 0]),
                    (f"{key}_v_px", clip.pixels[key][:, 1])]
    columns += [("volume_m3", clip.volume),
                ("metrics_h_m", height),
                ("metrics_cx_px", pixels[:, 0]),
                ("metrics_cy_px", pixels[:, 1])]
    return T.write_table(path, columns, digits=6)


def build_meta(clip, paths, check=None, smooth=None, flight=None):
    """写进 JSON 的全部口径与统计。下游脚本靠它找回源数据。

    `flight` 应在贝塞尔平滑**之前**算好再传入：腾空 g 是对原始曲线的物理校验，
    平滑后再拟合会把"看起来更接近 9.81"误当成链路更准。
    """
    events = clip.events()
    return {
        "source": {key: str(value) for key, value in paths.items()},
        "unit": "m", "up_axis": "Y",
        "origin": "画面中心所在的水平线（不是水面：单目标定给不出水面）",
        "fps": clip.fps,
        "frames": [int(clip.frames[0]), int(clip.frames[-1])],
        "usable_frames": int((clip.valid == 1).sum()),
        "modes": {key: A.MODE_DOC[key] for key in clip.modes},
        "alignment": {
            "method": "fixed depth plane; X/Y from the true source-video "
                      "projection of each vertex, Z carries body thickness only",
            "scale_mode": clip.scale_mode,
            "scale_m_per_px": round(clip.scale, 8),
            "scale_candidates": {k: (round(v, 8) if np.isfinite(v) else None)
                                 for k, v in clip.scale_table.items()},
            "body_height_m": round(clip.body_height(), 4),
        },
        "divf_header": {k: int(v) for k, v in clip.header.items()},
        "topology": clip.topology(),
        "events": {k: (int(v) if v is not None else None)
                   for k, v in events.items()},
        "flight": flight if flight is not None else clip.fit_flight(),
        "smooth": smooth or {"enabled": False},
        "reprojection": check,
        "gb": clip.gb_facts(),
    }
