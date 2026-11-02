"""整体旋转修复的前后对比视频：`spin` 产物 vs 它的输入（`retarget/mesh.fbx`）。

与 `video`（重心高度四格）看的东西不同，这里看的是**整体朝向**：
  - 相机跟着被盯质心平移、不转、固定比例（格高 2.8 m）。人占满格子，转动才看得清；
    相机不转，所以腾空中角动量守恒 = 箭头在画面里不动。
  - 三个视角：侧视（相机 +Z，主视角）/ 正视（相机 −X）/ 俯视（相机 +Y）。生硬感在后两个。
  - 琥珀箭头 = 全身角动量（1 cm / kg·m²/s），从质心画出。触床时变暗（床在施力）。
  - 只在整数帧求值。原文件 Root_M 的 cubic Euler 在键之间会甩整圈，25 fps 播放时根本
    求不到那些位置；按子帧渲染会让 before 凭空多出整圈突变——第一版就这么冤枉了原版。
  - 底部三条曲线随播放滚动：侧向轴 vs 世界 Z（转体）/ 离面 |L| / 空翻分量 L∥。

行 = before / after，列 = 三视角。
"""
import cv2
import numpy as np

from ..common import paths as P
from ..common import plotstyle as PS
from ..common import raster as R
from . import curves as Cv
from . import scene as F
from . import spin
from . import stage as St

CELL = 340                 # 格子边长 px
HALF_SPAN = 140.0          # 格子半高 cm：人伸直 1.8 m，留出箭头的余量
STRIP_H = 300
WINDOW = 40                # 曲线条左右各看多少帧
ARROW = (60, 190, 250)     # BGR 琥珀：不与人体肤色、两种口径色撞
# (标签, h 轴, h 符号, v 轴, v 符号, 深度轴, 深度符号)。正视/侧视与 `stage.VIEWS` 同向。
VIEWS = (("侧视 · 相机 +Z（主视角）", 0, +1, 1, +1, 2, +1),
         ("正视 · 相机 -X", 2, +1, 1, +1, 0, -1),
         ("俯视 · 相机 +Y", 0, +1, 2, -1, 1, +1))


class TrackCamera:
    """轴对齐正交相机，跟踪一个中心点。接口与 `raster.OrthoCamera` 兼容
    （project / project_one / light / depth_axis / depth_sign）。

    `OrthoCamera` 写死了"画面竖直 = 世界 Y"，俯视做不了，所以另写一个。
    """

    def __init__(self, h_axis, h_sign, v_axis, v_sign, depth_axis, depth_sign,
                 centre, size=CELL, half=HALF_SPAN):
        self.h, self.hs, self.v, self.vs = h_axis, h_sign, v_axis, v_sign
        self.depth_axis, self.depth_sign = depth_axis, depth_sign
        self.centre = np.asarray(centre, dtype=np.float64)
        self.width = self.height = size
        self.scale = (size / 2) / half
        toward, up, right = np.zeros(3), np.zeros(3), np.zeros(3)
        toward[depth_axis], up[v_axis], right[h_axis] = depth_sign, v_sign, h_sign
        light = (R.LIGHT_TOWARD_CAMERA * toward + R.LIGHT_UP * up
                 + R.LIGHT_SIDE * right)
        self.light = light / np.linalg.norm(light)

    def project(self, points):
        p = np.atleast_2d(points) - self.centre
        return np.column_stack([
            self.width / 2 + self.hs * p[:, self.h] * self.scale,
            self.height / 2 - self.vs * p[:, self.v] * self.scale,
        ]), -self.depth_sign * p[:, self.depth_axis]

    def project_one(self, point):
        return self.project(np.reshape(point, (1, 3)))[0][0]


def bgr(hexstr):
    value = hexstr.lstrip("#")
    r, g, b = (int(value[k:k + 2], 16) for k in (0, 2, 4))
    return (b, g, r)


def split_momentum(L, frames, windows):
    """(L∥, |L⊥|)：按 before 每段腾空的平均 L 定空翻轴，触床帧 nan。"""
    along = {k: np.full(len(frames), np.nan) for k in L}
    off = {k: np.full(len(frames), np.nan) for k in L}
    for start, stop in windows:
        a, b = int(np.searchsorted(frames, start)), int(np.searchsorted(frames, stop))
        axis = L["before"][a:b + 1].mean(0)
        axis /= np.linalg.norm(axis)
        plane = spin.plane_basis(axis)
        for key, series in L.items():
            along[key][a:b + 1] = np.abs(series[a:b + 1] @ axis)
            off[key][a:b + 1] = np.linalg.norm(series[a:b + 1] @ plane.T, axis=1)
    return along, off


class Strip:
    """底部滚动曲线条。matplotlib 只建一次，逐帧改 xlim 与播放头再 draw。"""

    def __init__(self, data, width, colours, labels):
        from ..common.plotstyle import plt

        self.width = width
        frames = data["frames"]
        windows = [tuple(w) for w in data["windows"]]
        L = {"before": data["L_before"], "after": data["L_after"]}
        along, off = split_momentum(L, frames, windows)
        series = ({"before": data["tilt_before"], "after": data["tilt_after"]},
                  off, along)
        titles = ("侧向轴 vs 世界 Z (°)：±90 = 纯空翻，过 0 = 在转体",
                  "离面角动量 |L⊥| (kg·m²/s)：守恒 = 水平线",
                  "空翻角动量 L∥ (kg·m²/s)：守恒 = 水平线")
        self.figure, self.axes = plt.subplots(
            1, 3, figsize=(width / 100, STRIP_H / 100), dpi=100)
        self.figure.patch.set_facecolor("#16181a")
        self.heads = []
        contacts = [(b0, a1) for (_a0, b0), (a1, _b1) in zip(windows[:-1], windows[1:])]
        for axis, title, values in zip(self.axes, titles, series):
            axis.set_facecolor("#16181a")
            for side in axis.spines.values():
                side.set_color("#55595e")
            axis.tick_params(colors="#a8aaac", labelsize=8)
            axis.set_title(title, color="#d8d8d8", fontsize=9, loc="left")
            axis.grid(alpha=0.15, lw=0.6)
            for a, b in contacts:
                axis.axvspan(a, b, color=PS.CONTACT, alpha=0.12, lw=0)
            for key in ("before", "after"):
                axis.plot(frames, values[key], "--" if key == "before" else "-",
                          c=colours[key], lw=1.3 if key == "before" else 1.6,
                          label=labels[key])
            self.heads.append(axis.axvline(frames[0], c="#e8e8e8", lw=1))
        self.axes[0].set_ylim(-95, 95)
        for axis, values in zip(self.axes[1:], series[1:]):
            top = np.nanmax(np.concatenate([values["before"], values["after"]]))
            axis.set_ylim(0, top * 1.08)
        self.axes[0].legend(loc="lower left", fontsize=7, frameon=False,
                            labelcolor="#d8d8d8", ncol=2)
        self.figure.tight_layout(pad=0.6)

    def render(self, frame):
        for axis, head in zip(self.axes, self.heads):
            axis.set_xlim(frame - WINDOW, frame + WINDOW)
            head.set_xdata([frame, frame])
        self.figure.canvas.draw()
        image = np.asarray(self.figure.canvas.buffer_rgba())[:, :, 2::-1]
        image = np.ascontiguousarray(image)
        # figsize 换算像素有 ±1 的舍入，拉回格子总宽才能 vstack
        if image.shape[:2] != (STRIP_H, self.width):
            image = cv2.resize(image, (self.width, STRIP_H))
        return image

    def close(self):
        from ..common.plotstyle import plt

        plt.close(self.figure)


def draw_cell(stage, stamp, view, centre, momentum, flying, painter, title, colour):
    label, h, hs, v, vs, d, ds = view
    camera = TrackCamera(h, hs, v, vs, d, ds, centre)
    canvas = np.full((CELL, CELL, 3), R.BACKGROUND, np.uint8)
    for vertices, faces, tint in stage.layers(stamp):
        R.draw_mesh(canvas, camera, vertices, faces, tint)
    # 箭头单位：1 cm / (kg·m²/s)。场景是厘米，L 是 SI，数值上直接当厘米长度画。
    tail = camera.project_one(centre)
    head = camera.project_one(centre + momentum)
    shade = ARROW if flying else R.fade(ARROW, 0.45, R.BACKGROUND)
    cv2.arrowedLine(canvas, tuple(int(x) for x in tail), tuple(int(x) for x in head),
                    shade, 3, cv2.LINE_AA, tipLength=0.18)
    cv2.circle(canvas, (int(tail[0]), int(tail[1])), 5, (250, 250, 250), -1,
               cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (CELL - 1, CELL - 1), (64, 62, 60), 1)
    cv2.rectangle(canvas, (0, 0), (5, 30), colour, -1)
    painter.paint(canvas, [(12, 6, title, (232, 232, 232), "small"),
                           (12, CELL - 26, label, (170, 168, 166), "small")])
    return canvas


def run(args, before_path, after_path, data_path, out_path):
    for path in (before_path, after_path, data_path):
        if not path.is_file():
            raise SystemExit(f"先跑 spin：需要 {P.display(path)}")
    data = np.load(data_path)
    frames, theta = data["frames"], data["theta"]
    windows = [tuple(w) for w in data["windows"]]
    centre = data["com"]
    first, last = args.frames or (int(frames[0]), int(frames[-1]))
    picked = np.flatnonzero((frames >= first) & (frames <= last))
    colours = {"before": PS.BEFORE, "after": PS.COLOR["spin"]}
    labels = {"before": "before · retarget", "after": "after · spin"}
    width = CELL * len(VIEWS)
    PS.setup()
    strip = Strip(data, width, colours, labels)
    painter = R.TextPainter()
    frames_dir = out_path.parent / f"frames_{out_path.stem}"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for stale in frames_dir.glob("*.png"):
        stale.unlink()

    opened = []
    try:
        stages = {}
        for key, path in (("before", before_path), ("after", after_path)):
            manager, scene = F.load(path)
            opened.append(manager)
            sampler = Cv.Sampler(scene, ("mesh",), args.body, args.bed,
                                 args.root_bone)
            stages[key] = (sampler, St.Stage(sampler))
        momentum = {"before": data["L_before"], "after": data["L_after"]}
        print(f"{after_path.name}: 渲染 f{first}..{last}（{len(picked)} 帧），"
              f"{width}×{38 + 2 * CELL + STRIP_H}")
        for count, index in enumerate(picked):
            frame = int(frames[index])
            flying = any(a < frame < b for a, b in windows)
            correction = np.degrees(np.linalg.norm(theta[index]))
            rows = []
            for key in ("before", "after"):
                sampler, stage = stages[key]
                stamp = sampler.stamp(frame)
                title = (labels[key] if key == "before"
                         else f"{labels[key]}   修正 {correction:5.1f}°")
                rows.append(np.hstack([
                    draw_cell(stage, stamp, view, centre[index], momentum[key][index],
                              flying, painter, title, bgr(colours[key]))
                    for view in VIEWS]))
            text = (f"f{frame} / {int(frames[-1])}   {'腾空' if flying else '触床'}"
                    f"   琥珀箭头 = 全身角动量（腾空中守恒 = 箭头不动）")
            image = np.vstack([R.banner(width, painter, text)] + rows
                              + [strip.render(frame)])
            cv2.imwrite(str(frames_dir / f"{count + 1:04d}.png"), image)
            if (count + 1) % 100 == 0 or count + 1 == len(picked):
                print(f"    {count + 1}/{len(picked)}")
    finally:
        strip.close()
        for manager in reversed(opened):
            manager.Destroy()
    video = R.encode(frames_dir, out_path, float(data["fps"]), keep=args.keep_frames)
    print(f"  wrote {P.display(video)} ({video.stat().st_size / 1e6:.1f} MB)")
