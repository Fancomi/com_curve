"""固定深度平面对齐与尺度标定。**这条线所有"为什么"都在这里。**

核心是一个问题：frames.bin 里每帧只有 crop 相机（focal=5000, patch=256, bbox, cam_t）。
cam_t.z 是单目深度估计，实测这段片子从 51 m 抖到 110 m（帧间最大跳 24 m）。照
dive-replay-3d 的 video 模式把顶点搬到世界里，刚体平移就带着这份抖动——叠在源视频上
看不出来（眼睛对齐的是底图里的人），去掉底图做正交渲染，整个人就在横向滑：7 s 附近
单帧横移 0.44 m。

解法：**固定深度平面**。选一个常数 Z0，人放在这个平面上：

    world_X = (u - W/2) * s
    world_Y = -(v - H/2) * s        （图像 y 向下，取负得到 Y 向上）
    s = Z0 / focal                  米每像素，全片常数

u/v 是**该顶点在源视频里的真实投影**（用每帧自己的 crop 相机算，与 overlay 看到的
一致）。于是重投影 u/v 逐位还原，"3D global 渲染与 2D 投影一致"是构造保证的，不是
拟合出来的。抖动被隔离在深度方向——而深度本来就是单目拿不到的量，侧视图因此只有
体厚、没有前后位移，这是诚实的表达。

s 的三个来源（`--scale`）：

  hsmr     每帧 tz/fx 的中位数 = 0.010804。与 HSMR 自己的体尺自洽（mesh 1.77 m 高、
           投影 154 px，比值对得上），但腾空段 g = -12.1。
  metrics  metrics.json 的 pixel_to_meter_ratio = 0.008157，来自分析器
           ref_distance_m=3.0 的标定。独立于 HSMR，腾空段 g = -9.13 —— 差 7%，
           这是对整条链路的独立校验，所以拿它当默认。
  gravity  反解出让腾空段恰好 -9.81 的 s = 0.008758。用它报 g 是循环论证，只作对照。

三者相差约 25%：HSMR 的绝对深度与图像标定不一致。一个标量 s 同时决定体尺和轨迹幅度，
两者不可能都对——这是数据本身的矛盾，不是这里的近似。
"""
import numpy as np

from ..common import geometry as G
from ..common import signal as Sig
from . import divf
from . import landmarks as Lm

SOURCE_W = 1920.0
SOURCE_H = 1080.0
GRAVITY = 9.81
SCALE_CHOICES = ("metrics", "hsmr", "gravity")
MODES = ("mesh", "gb")
MODE_DOC = {
    "mesh": "homogeneous volume centroid of the skin mesh, per frame",
    "gb": "GB/T 17245-2004 15-segment model: masses from Table 6, centroids at "
          "Table 7 ratios on hand-labelled surface bony landmarks",
}
# 抛物线拟合之后、渲染之前的轻度贝塞尔重采样。`n_ctrl` 是"约 100 帧时的控制点数"，
# 更长的连续有效段会按比例加控制点，避免 8 个点硬套 400 帧把整段弧抹平。
DEFAULT_BEZIER_CTRL = 12


def project_source(local, record):
    """HSMR local 顶点 -> 源视频像素 (u, v)。

    HSMR 在 patch×patch 的 crop 上以 focal 出图，crop 再线性映回 bbox：

        u_crop = patch/2 + focal * (x + tx) / (z + tz)
        u_src  = bbox.x + u_crop * bbox.w / patch

    这就是 overlay 里看到的那套投影，逐位一致。
    """
    bbox, focal, patch = record["bbox"], record["focal"], record["patch"]
    tx, ty, tz = record["cam"]
    points = np.atleast_2d(local)
    depth = points[:, 2] + tz
    depth = np.where(np.abs(depth) < 1e-9, np.nan, depth)
    out = np.column_stack([
        bbox[0] + (patch * 0.5 + focal * (points[:, 0] + tx) / depth)
        * bbox[2] / patch,
        bbox[1] + (patch * 0.5 + focal * (points[:, 1] + ty) / depth)
        * bbox[3] / patch,
    ])
    return out[0] if np.ndim(local) == 1 else out


def place_on_plane(local, record, scale, scale_hsmr):
    """一帧顶点 -> 固定深度平面上的 global 坐标（米，Y 向上）。

        X    =  (u - W/2) * s
        Y_up = -(v - H/2) * s
        Z    =  local_z 去中位数后 × (s / s_hsmr)

    X/Y 完全由 (u, v) 决定，所以重投影逐位还原。Z 只承载体厚：HSMR local 的单位是
    它自己那套米（s_hsmr 米每像素），换算到 s 这套单位要乘 s/s_hsmr，人才是等比缩放
    而不是被压扁。单目拿不到真实前后位移，Z 里因此**不含轨迹**。
    """
    pixels = project_source(local, record)
    ratio = scale / scale_hsmr if scale_hsmr > 1e-12 else 1.0
    return np.column_stack([
        (pixels[:, 0] - SOURCE_W * 0.5) * scale,
        -(pixels[:, 1] - SOURCE_H * 0.5) * scale,
        (local[:, 2] - np.median(local[:, 2])) * ratio,
    ])


def hsmr_scale(records):
    """每帧 tz / fx_src 的中位数。fx_src = focal * bbox.w / patch。"""
    values = [r["cam"][2] / (r["focal"] * r["bbox"][2] / r["patch"])
              for r in records.values()
              if r["focal"] * r["bbox"][2] / r["patch"] > 1e-6]
    if not values:
        raise SystemExit("无法估计 hsmr 尺度：没有有效帧")
    return float(np.median(values))


def gravity_scale(measured, reference):
    """让腾空段拟合出 -9.81 的 s。一步闭式，不需要迭代。

    `place_on_plane` 是关于原点的**均匀缩放**：X、Y 都是像素乘 s，Z 是 local_z 乘
    s/s_hsmr。所以整块几何按 s 等比放大，体积重心随之等比，高度曲线 Y(s) = s·Y(1)，
    拟合出的加速度 g(s) = s·g(1)。于是 s* = s_ref · 9.81 / |g(s_ref)|。

    先在像素上解一次再套用是不对的——透视投影非线性，"重心的投影"不等于"投影的重心"，
    那样解出来会落在 -10.2 而不是 -9.81。必须拿真正的体积重心量一次 g 再按比例缩。
    """
    if not np.isfinite(measured) or abs(measured) < 1e-9:
        return float("nan")
    return float(reference * GRAVITY / abs(measured))


def _fill_gaps(series):
    """线性插值补上 nan，全 nan 时原样返回。拟合前用。"""
    mask = np.isfinite(series)
    if mask.all() or not mask.any():
        return series, mask
    filled = series.copy()
    index = np.arange(len(series))
    filled[~mask] = np.interp(index[~mask], index[mask], series[mask])
    return filled, mask


class Clip:
    """一段 replay：解析、定尺度、放到平面上、算两条曲线。

    两遍扫：第一遍存 record 与像素投影（与 s 无关），据此定 s；第二遍按 s 放到平面上
    做体积积分。这样 `--scale gravity` 能在同一条路径里线性解出来，不用先猜后迭。
    """

    def __init__(self, frames_path, skin_obj, metrics, scale_mode="metrics",
                 fps=None, trim=8, verbose=True, landmarks=None):
        raw = divf.read(frames_path)
        self.header = divf.header(raw)
        self.rest, self.faces = divf.load_obj(skin_obj)
        self.metrics = metrics or {}
        self.fps = float(fps or self.metrics.get("fps")
                         or self.header["fps"] or 60.0)
        self.trim = trim
        self.count = self.header["n"]
        # gb 口径要读每帧的 skel 块（国标骨突多标在骨骼上）；没给标注就不采这条。
        self.landmarks = landmarks
        self.sampler = (Lm.GbSampler(landmarks) if landmarks is not None else None)
        self.modes = MODES if self.sampler is not None else tuple(
            k for k in MODES if k != "gb")

        self.records = {}
        for index in range(self.count):
            record = divf.record(raw, self.header, index,
                                 want_skel=self.sampler is not None)
            if record["usable"]:
                self.records[index] = record
        if not self.records:
            raise SystemExit("没有可用的 mesh 帧")

        self.frames = np.arange(self.count, dtype=np.int32)
        self.status = np.array([self.records[i]["status"] if i in self.records else 0
                                for i in range(self.count)], np.int32)
        self.local = {key: np.full((self.count, 3), np.nan) for key in self.modes}
        self.pixels = {key: np.full((self.count, 2), np.nan) for key in self.modes}
        for index, record in self.records.items():
            skin = record["skin"]
            try:
                _volume, centre = G.volume_centroid(skin, self.faces)
            except ValueError:
                centre = skin.mean(0)
            self.local["mesh"][index] = centre
            if self.sampler is not None:
                self.local["gb"][index] = self.sampler.centre(
                    {"skin": skin, "skel": record["skel"]})
            for key in self.modes:
                self.pixels[key][index] = project_source(self.local[key][index],
                                                         record)

        self.flight = self._flight_window()
        self.scale_hsmr = hsmr_scale(self.records)
        self.scale_table = {
            "hsmr": self.scale_hsmr,
            "metrics": float(self.metrics.get("pixel_to_meter_ratio") or 0.0)
            or float("nan"),
            "gravity": float("nan"),
        }
        if self.flight is not None:
            probe = self._place(self.scale_hsmr)
            self.scale_table["gravity"] = gravity_scale(
                self._flight_gravity(probe["mesh"]), self.scale_hsmr)

        self.scale_mode = scale_mode
        self.scale = self.scale_table.get(scale_mode)
        if self.scale is None or not np.isfinite(self.scale) or self.scale <= 0:
            # 无 metrics 时 metrics 尺度不可用；自动退到 hsmr，避免整批任务直接退出。
            fallback = next((k for k in ("hsmr", "gravity")
                             if np.isfinite(self.scale_table.get(k, float("nan")))
                             and self.scale_table[k] > 0), None)
            if fallback is None:
                raise SystemExit(f"尺度 {scale_mode} 不可用：{self.scale_table}")
            if verbose:
                print(f"  尺度 {scale_mode} 不可用，改用 {fallback}")
            self.scale_mode = fallback
            self.scale = self.scale_table[fallback]
        if verbose:
            print(f"  尺度({self.scale_mode}) = {self.scale:.6f} m/px   候选 "
                  + "  ".join(f"{k}={v:.6f}" for k, v in self.scale_table.items()
                              if np.isfinite(v)))
        self.__dict__.update(self._place(self.scale))

    # ---- 内部 ----------------------------------------------------------

    def _flight_window(self):
        """腾空段（离板 -> 入水）。优先 metrics 事件，再收窄到含最高点的一条弧。

        有 metrics 时：不自己猜 jump/water（上游分析器已给事件帧），只把
        jumping_frame..water_entry_frame 收窄到 peak 两侧各自最低点——20260801
        那种多次弹跳整段拟会得到 g≈-1.16，收窄后才接近 -9。

        无 metrics 时（新下载包常见）：用 mesh 像素高度自检——取最长连续有效段，
        在其中找最高点，再取两侧谷底。这比没有窗口强，但不如分析器事件稳，
        结果写进 JSON 的 events 时 peak 有值、jump/water 可能为空。
        """
        height = -self.pixels["mesh"][:, 1]     # 图像 y 向下，取负得到"越高越大"
        summary = self.metrics.get("summary") or {}
        jump, water = summary.get("jumping_frame"), summary.get("water_entry_frame")
        peak = summary.get("peak_frame")
        if jump is not None and water is not None and int(water) > int(jump) + 20:
            low, high = int(jump), int(water)
            if peak is None or not low < int(peak) < high:
                return (low, high)
            peak = int(peak)
            return (low + int(np.nanargmin(height[low:peak + 1])),
                    peak + int(np.nanargmin(height[peak:high + 1])))

        # ---- 无 metrics 回退：最长连续有效段 + 峰两侧谷底 ----
        valid = np.isfinite(height)
        runs = Sig.runs_of(valid)
        if not runs:
            return None
        low, high = max(runs, key=lambda ab: ab[1] - ab[0])
        if high - low < 20:
            return None
        peak = low + int(np.nanargmax(height[low:high + 1]))
        if peak <= low or peak >= high:
            return (low, high)
        left = low + int(np.nanargmin(height[low:peak + 1]))
        right = peak + int(np.nanargmin(height[peak:high + 1]))
        if right - left < 20:
            return (low, high)
        return (left, right)

    def _place(self, scale):
        """按给定 s 放置全片。返回 {meshes, mesh, gb, volume, valid}（gb 仅在有标注时）。"""
        out = {key: np.full((self.count, 3), np.nan) for key in self.modes}
        out.update({"volume": np.full(self.count, np.nan),
                    "valid": np.zeros(self.count, np.int8),
                    "meshes": [None] * self.count})
        for index, record in self.records.items():
            placed = place_on_plane(record["skin"], record, scale, self.scale_hsmr)
            try:
                out["volume"][index], out["mesh"][index] = G.volume_centroid(
                    placed, self.faces)
            except ValueError:
                out["mesh"][index] = placed.mean(0)
            if self.sampler is not None:
                # 骨骼层也要放到同一个平面上：`place_on_plane` 用的是**该顶点自己**
                # 在源视频里的投影，所以两层各放一次，结果自然同一坐标系。
                placed_skel = place_on_plane(record["skel"], record, scale,
                                             self.scale_hsmr)
                out["gb"][index] = self.sampler.centre(
                    {"skin": placed, "skel": placed_skel})
            out["meshes"][index] = placed
            out["valid"][index] = 1
        return out

    def _flight_gravity(self, track):
        """腾空段切端后的拟合加速度。窗口与 `fit_flight` 报的那个一致。"""
        span = np.arange(self.flight[0], self.flight[1] + 1)
        height, mask = _fill_gaps(track[span, 1])
        if mask.sum() < 2 * self.trim + 3:
            return float("nan")
        return Sig.fit_parabola(height[self.trim:len(height) - self.trim],
                                1.0 / self.fps)[0]

    # ---- 对外 ----------------------------------------------------------

    def body_height(self):
        first = next((m for m in self.meshes if m is not None), None)
        return (float(first[:, 1].max() - first[:, 1].min())
                if first is not None else float("nan"))

    def reprojection_error(self):
        """把 global 顶点重投影回像素，与第一遍的投影比。

        这是"3D global 渲染与 2D 投影一致"的可执行形式。理论上应当是 0；实测残差来自
        float32 顶点的舍入，量级 1e-2 px。**不查重心**——透视投影非线性，"重心的投影"
        不等于"投影的重心"，那个差是真实的几何事实，不是 bug。
        """
        worst, total, count = 0.0, 0.0, 0
        for index, record in self.records.items():
            placed = self.meshes[index]
            if placed is None:
                continue
            u = SOURCE_W * 0.5 + placed[:, 0] / self.scale
            v = SOURCE_H * 0.5 - placed[:, 1] / self.scale
            truth = project_source(record["skin"], record)
            error = np.hypot(u - truth[:, 0], v - truth[:, 1])
            worst = max(worst, float(error.max()))
            total += float(error.sum())
            count += len(error)
        return ({"count": count, "mean_px": total / count, "max_px": worst}
                if count else {"count": 0})

    def fit_flight(self):
        """腾空段抛物线拟合。单位是米，所以 g 可以直接和 -9.81 比。"""
        out = {"trim_frames": self.trim, "flight": list(self.flight or []),
               "per_curve": {}}
        if self.flight is None:
            return out
        span = np.arange(self.flight[0], self.flight[1] + 1)
        dt = 1.0 / self.fps
        for key in self.modes:
            series = getattr(self, key)[span, 1]
            height, mask = _fill_gaps(series)
            if mask.sum() < 5:
                continue
            gravity, rms = Sig.fit_parabola(height, dt)
            core = height[self.trim:len(height) - self.trim]
            trimmed = (Sig.fit_parabola(core, dt) if len(core) >= 3
                       else (float("nan"), float("nan")))
            out["per_curve"][key] = {
                "gravity_mps2": round(float(gravity), 3),
                "fit_rms_m": round(float(rms), 4),
                "gravity_trimmed_mps2": round(float(trimmed[0]), 3),
                "fit_rms_trimmed_m": round(float(trimmed[1]), 4),
                "apex_m": round(float(np.nanmax(series)), 4),
                "rise_m": round(float(np.nanmax(series) - series[mask][0]), 4),
            }
        return out

    def topology(self):
        closed, edges, max_shared = G.is_closed(self.faces)
        return {"closed_manifold": closed, "edges": edges,
                "max_faces_per_edge": max_shared, "triangles": int(len(self.faces))}

    def gb_facts(self):
        """gb 口径的标注与分段事实。没采这条曲线时返回 None。

        段长与国标反解的占身高比例并排列出——这是「端点接错」唯一能自动抓住的判据。
        用第一个有效帧的 local 顶点（未放到平面上）：那是 HSMR 自己那套米，段长比例
        与 s 无关，不受尺度口径影响。
        """
        if self.sampler is None:
            return None
        index = next(iter(self.records))
        record = self.records[index]
        return {**self.landmarks.facts(),
                **self.sampler.facts({"skin": record["skin"],
                                 "skel": record["skel"]}),
                "reference_frame": int(index)}

    def events(self):
        """metrics.json 里的四个事件帧。"""
        summary = self.metrics.get("summary") or {}
        return {key: summary.get(key) for key in
                ("board_frame", "jumping_frame", "peak_frame",
                 "water_entry_frame")}

    def smooth_curves(self, n_ctrl=DEFAULT_BEZIER_CTRL):
        """抛物线拟合之后、渲染之前：对 `modes` 里每条曲线做轻度贝塞尔重采样。

        只动曲线，不动 `meshes`——人体姿态仍是原始放置结果；变的是轨迹线、高度
        读数和 overlay 面板。先 `fit_flight()` 再调本方法：腾空 g 仍用未平滑曲线，
        避免"平滑后更好看所以 g 更接近 9.81"的自欺。返回写入 JSON 的参数摘要。
        """
        n_ctrl = int(n_ctrl)
        if n_ctrl < 2:
            raise ValueError(f"贝塞尔控制点数至少 2，给了 {n_ctrl}")
        valid = self.valid == 1
        stats = {}
        for key in self.modes:
            original = getattr(self, key)
            smoothed = Sig.smooth_track(original, valid, n_ctrl=n_ctrl)
            good = valid & np.isfinite(original).all(1) & np.isfinite(smoothed).all(1)
            delta = (np.abs(smoothed[good] - original[good]) if good.any()
                     else np.zeros((0, 3)))
            setattr(self, key, smoothed)
            stats[key] = {
                "max_delta_m": [round(float(v), 5) for v in
                                (delta.max(0) if len(delta) else [0, 0, 0])],
                "mean_delta_m": [round(float(v), 5) for v in
                                 (delta.mean(0) if len(delta) else [0, 0, 0])],
            }
        return {"enabled": True, "n_ctrl": n_ctrl, "method":
                "least-squares Bezier resample on contiguous valid runs "
                "(sport_detect BezierTools-compatible, pure numpy)",
                "per_curve": stats}
