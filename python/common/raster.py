"""软件光栅器：正交投影 + 画家算法 + 背面剔除。无 GPU 依赖。

三个渲染器（trampoline 四格、dive 四格、dive 叠源视频）共用这一份。它们的差别只在
**相机**和**要画什么**，光栅化与标记绘制完全相同。

用 `cv2.fillConvexPoly` 而不是自己写 numpy 扫描线——实测前者 8 ms 画完一帧的 6000 个
三角形，后者 212 ms，差 26 倍。`shift=4` 走 1/16 像素的定点坐标，配合 LINE_AA 消掉
三角形之间的接缝（整数坐标下相邻面会露出背景色的细线）。

单位无关：`OrthoCamera` 只认"世界单位"，trampoline 传厘米、dive 传米，刻度标签的
单位名由调用方给。
"""
import subprocess

import cv2
import numpy as np

BACKGROUND = (26, 24, 22)
# 光源是**每台相机各一个头灯**，不是共用的世界方向。共用世界光源试过：光偏 +Z 时
# 侧视格明亮而正视格只剩环境项，人整个是灰的——正视能看见的面法向朝 −X，与光几乎
# 垂直。每台相机自带一个偏左上的头灯，两格亮度才一致，且明暗仍随姿态变化。
LIGHT_TOWARD_CAMERA = 0.72
LIGHT_UP = 0.52
LIGHT_SIDE = -0.34
# 0.28 的环境项：纯 Lambert 会让背光面全黑，人体轮廓在深色背景上就断掉了。
AMBIENT = 0.28


def _headlight(axis, depth_axis, depth_sign):
    light = np.zeros(3)
    light[depth_axis] = LIGHT_TOWARD_CAMERA * depth_sign
    light[1] = LIGHT_UP
    light[axis] = LIGHT_SIDE
    return light / np.linalg.norm(light)


class OrthoCamera:
    """正交相机。比例由垂直范围与画面高度决定，宽度按水平跨度算。

    `min_width` 保证标题放得下：标签在 23 px 字号下约 200 px 宽，窄于这个数的格子
    会把标题裁掉右半截——渲染短片段调试时最容易碰到，因为世界跨度是从所选帧算的。
    宽度取偶数：yuv420p 的色度 2×2 下采样要求两个维度都是偶数。
    """

    def __init__(self, axis, depth_axis, depth_sign, x_range, y_range, height,
                 min_width=232):
        self.axis, self.depth_axis, self.depth_sign = axis, depth_axis, depth_sign
        self.height = height
        self.y_low, self.y_high = y_range
        self.scale = (height - 1) / (self.y_high - self.y_low)
        width = int(round((x_range[1] - x_range[0]) * self.scale)) // 2 * 2
        self.width = max(width, min_width // 2 * 2)
        # 补宽往两侧平均加，否则内容会贴着左边。
        pad = (self.width - width) / 2.0 / self.scale
        self.x_low = x_range[0] - pad
        self.light = _headlight(axis, depth_axis, depth_sign)

    def project(self, points):
        """(N, 2) 像素坐标 + (N,) 深度。深度越小越靠近相机。"""
        points = np.atleast_2d(points)
        return np.column_stack([
            (points[:, self.axis] - self.x_low) * self.scale,
            (self.height - 1) - (points[:, 1] - self.y_low) * self.scale,
        ]), -self.depth_sign * points[:, self.depth_axis]

    def project_one(self, point):
        return self.project(np.reshape(point, (1, 3)))[0][0]

    def row_of(self, value):
        """世界高度对应的像素行号。"""
        return int(round((self.height - 1)
                         - (value - self.y_low) * self.scale))

    def visible_rows(self, step=1.0):
        """画面里可见的整数刻度值（按 `step` 取整）。"""
        low = int(np.floor(self.y_low / step))
        high = int(np.ceil(self.y_high / step))
        return [k * step for k in range(low, high + 1)]


class PixelCamera:
    """global（Y 向上）-> 源视频像素。固定深度平面映射的闭式逆。

    只有一个标量 scale，所以这不是"另一台相机"——它就是原来那台机位。任何在
    global 里算出来的点（质心、标尺刻度、轨迹）都能无损搬回像素。
    """

    def __init__(self, scale, width, height):
        self.scale, self.width, self.height = scale, width, height
        # global Z 越大越远离相机，画家算法要远到近，所以按 Z 降序画。
        self.axis, self.depth_axis, self.depth_sign = 0, 2, -1
        self.light = _headlight(self.axis, self.depth_axis, self.depth_sign)

    def project(self, points):
        points = np.atleast_2d(points)
        return np.column_stack([
            self.width * 0.5 + points[:, 0] / self.scale,
            self.height * 0.5 - points[:, 1] / self.scale,
        ]), -self.depth_sign * points[:, self.depth_axis]

    def project_one(self, point):
        return self.project(np.reshape(point, (1, 3)))[0][0]

    def row_of(self, value):
        return int(round(self.height * 0.5 - value / self.scale))

    def visible_rows(self, step=1.0):
        span = self.height * 0.5 * self.scale
        low, high = int(np.floor(-span / step)), int(np.ceil(span / step))
        return [k * step for k in range(low, high + 1)]


def shade(vertices, faces, tint, camera):
    """(每面颜色 BGR, 深度, 是否朝向相机)。

    法向由三角形绕向决定，而绕向已被体积计算验证过朝外（散度定理给出正体积），
    所以可以直接用 `normal · 相机方向 > 0` 做背面剔除。
    """
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    normals = np.cross(b - a, c - a)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.where(lengths < 1e-12, 1.0, lengths)
    facing = normals[:, camera.depth_axis] * camera.depth_sign > 0
    # errstate 屏蔽 Apple Accelerate 在长 matmul 上的假警告（输出逐位正确）。
    with np.errstate(all="ignore"):
        lambert = np.clip(normals @ camera.light, 0.0, 1.0)
    intensity = AMBIENT + (1.0 - AMBIENT) * lambert
    colours = np.clip(np.asarray(tint)[None, :] * intensity[:, None],
                      0, 255).astype(np.uint8)
    depth = (a[:, camera.depth_axis] + b[:, camera.depth_axis]
             + c[:, camera.depth_axis]) / 3.0 * -camera.depth_sign
    return colours, depth, facing


def fill_faces(canvas, pixels, faces, colours, depth, facing):
    """画家算法：远到近逐面填充。"""
    points = (pixels[faces] * 16.0).astype(np.int32)
    for index in np.argsort(-depth):
        if facing[index]:
            cv2.fillConvexPoly(canvas, points[index],
                               tuple(int(v) for v in colours[index]),
                               lineType=cv2.LINE_AA, shift=4)


def draw_mesh(canvas, camera, vertices, faces, tint, alpha=1.0, silhouette=None,
              rim_colour=(245, 190, 235)):
    """把一个网格画上画布。`alpha < 1` 时整体半透明混合。

    逐面 addWeighted 会在三角形重叠处叠出深斑（同一块皮肤被混了两次）——先在不透明
    副本上画完整个人，再一次混合，颜色才均匀。

    `silhouette` 是 `rim_edges` 的结果时额外描一圈**不参与混合**的轮廓：叠在源视频上
    时这个人只有 160 px 高，半透明填充在米白墙面上几乎融掉，描边给出与背景无关的
    确定边界。
    """
    colours, depth, facing = shade(vertices, faces, tint, camera)
    pixels, _depth = camera.project(vertices)
    target = canvas if alpha >= 1.0 else canvas.copy()
    fill_faces(target, pixels, faces, colours, depth, facing)
    if alpha < 1.0:
        cv2.addWeighted(target, alpha, canvas, 1.0 - alpha, 0.0, dst=canvas)
    if silhouette is not None:
        edges, owners = silhouette
        on_rim = facing[owners[:, 0]] != facing[owners[:, 1]]
        segments = (pixels[edges[on_rim]] * 16.0).astype(np.int32)
        cv2.polylines(canvas, segments, False, rim_colour, 1, cv2.LINE_AA, shift=4)
    return canvas


def rim_edges(faces):
    """(边, 两个邻面) —— 每帧用来找轮廓边（一面朝前一面朝后的那条）。

    拓扑全片不变，所以只建一次。要求闭合流形，边界边会被丢掉。
    """
    from .geometry import edge_owners

    pairs = [(e, o) for e, o in edge_owners(faces).items() if len(o) == 2]
    return (np.array([e for e, _o in pairs], np.int64),
            np.array([o for _e, o in pairs], np.int64))


def fade(colour, weight, background=None):
    """按 `weight` 淡化。给了 `background` 就朝它插值，否则乘性变暗。

    深色背景上朝背景插值最自然；叠在源视频上没有确定的背景色，只能乘性变暗。
    """
    if background is None:
        return tuple(int(c * weight) for c in colour)
    return tuple(int(background[k] + (colour[k] - background[k]) * weight)
                 for k in range(3))


def draw_trail(canvas, camera, track, upto, colour, span=45, dashed=False,
               thickness=2, background=BACKGROUND, faint=None):
    """重心的近期轨迹，越旧越淡；跨过无效帧时断开而不是连一条假线。

    `span` 一个完整腾空周期左右：再长就把上一次弹跳的弧线也拖进来，画面里同时出现
    两条弧反而看不出当前这一跳的形状。

    `faint` 给了就先把**全程**轨迹淡画一层：只画近期看不出这一跳整体的形状，只画
    全程则当前在哪不明显。

    `dashed` 隔段画。同一格里叠两条轨迹时靠线型区分（虚线是根骨骼，实线是重心），
    颜色不够用——它已经承载 before/after 的区别了。
    """
    ok = np.isfinite(track).all(axis=1)
    pixels, _depth = camera.project(np.nan_to_num(track, nan=0.0))
    points = pixels.astype(np.int32)
    last = min(upto, len(points) - 1)

    def link(index, shade_colour, width):
        if ok[index] and ok[index + 1]:
            cv2.line(canvas, tuple(points[index]), tuple(points[index + 1]),
                     shade_colour, width, cv2.LINE_AA)

    if faint is not None:
        for index in range(last):
            link(index, faint, 1)
    start = max(0, upto - span)
    count = max(last - start, 1)
    for index in range(start, last):
        if dashed and (index - start) % 3 == 2:
            continue
        weight = (index - start + 1) / count
        link(index, fade(colour, weight ** 1.6, background), thickness)


def draw_marker(canvas, camera, position, colour, hollow=False, style="disc",
                rule=True):
    """重心标记 + 一条贯穿画面的高度线。

    高度线是多格能互相对齐的那个东西——各格共用同一个比例，所以这条线的像素行号
    在各格里相同，两条曲线的高度差直接表现为相邻两格线位的错开。

    `style="disc"` 实心点 + 外环（深色背景上）；`style="ring"` 空心环 + 十字
    （叠在源视频上：实心圆会把腹部整块盖掉，而"重心落在身体哪个位置"恰好是要看的）。
    `hollow` 是同一格里的副标记（根骨骼），画得小一圈并配虚线高度线——两者相距常常
    不到 20 px，同样大小会叠成一团。
    """
    if not np.isfinite(position).all():
        return
    pixel = camera.project_one(position)
    centre = (int(round(pixel[0])), int(round(pixel[1])))
    width = canvas.shape[1]
    if rule and 0 <= centre[1] < canvas.shape[0]:
        dim = fade(colour, 0.34 if hollow else 0.5,
                   BACKGROUND if style == "disc" else None)
        if hollow:
            for x in range(0, width, 12):
                cv2.line(canvas, (x, centre[1]), (min(x + 6, width), centre[1]),
                         dim, 1, cv2.LINE_AA)
        else:
            cv2.line(canvas, (0, centre[1]), (width, centre[1]), dim, 1,
                     cv2.LINE_AA)
    if hollow:
        cv2.circle(canvas, centre, 6, colour, 2, cv2.LINE_AA)
        return
    if style == "ring":
        cv2.circle(canvas, centre, 9, (250, 250, 250), 3, cv2.LINE_AA)
        cv2.circle(canvas, centre, 9, colour, 2, cv2.LINE_AA)
        for dx, dy in ((14, 0), (0, 14)):
            cv2.line(canvas, (centre[0] - dx, centre[1] - dy),
                     (centre[0] + dx, centre[1] + dy), colour, 2, cv2.LINE_AA)
        return
    cv2.circle(canvas, centre, 9, (250, 250, 250), -1, cv2.LINE_AA)
    cv2.circle(canvas, centre, 7, colour, -1, cv2.LINE_AA)
    cv2.circle(canvas, centre, 13, colour, 2, cv2.LINE_AA)


def draw_discs(canvas, camera, centres, weights, colour, full_scale, radius=19,
               alpha=0.62):
    """一组圆，**面积**正比于 `weights`（除以 `full_scale` 归一）。

    面积而不是半径正比：半径正比会让躯干（49.7%）的圆比手（0.6%）大 80 倍，直径吃掉
    半格画面。面积正比下半径比是 √80 ≈ 9。

    `full_scale` 是各格共用的满刻度而不是各自的最大值——两种质量口径的分歧恰好就在
    "躯干占比 38% vs 50%"上，各自归一会把它擦掉。

    半透明：不透明的躯干圆直径约 40 px，会把胸腹盖掉，而"这一帧人是什么姿态"是理解
    重心为什么在那儿的前提。
    """
    pixels, depth = camera.project(centres)
    factor = radius / np.sqrt(full_scale)
    overlay = canvas.copy()
    for index in np.argsort(-depth):
        point = (int(round(pixels[index, 0])), int(round(pixels[index, 1])))
        size = max(2, int(round(factor * np.sqrt(weights[index]))))
        cv2.circle(overlay, point, size, colour, -1, cv2.LINE_AA)
        cv2.circle(overlay, point, size, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0.0, dst=canvas)


def draw_scale(canvas, camera, unit="m", step=1.0, label_step=None, top=100,
               datum=None, heavy_every=1, tick=26, font=0.42):
    """左侧高度标尺 + 可选的基准虚线。

    刻度按**世界单位**的 `step` 取整，标签按 `label_step` 换算后打印——trampoline 的
    世界单位是厘米而刻度要标"1m"，两者差 100 倍。`label_step` 默认等于 `step`，即
    标签就是刻度序号乘 1。

    `top` 是上边留给标签的行数，刻度不画进那片区域——否则会和"h = 4.60 m"叠在一起。
    `heavy_every` > 1 时只有整数倍刻度带数字（叠在源视频上刻度密，全标会糊）。
    """
    label_step = step if label_step is None else label_step
    for index, value in enumerate(camera.visible_rows(step)):
        row = camera.row_of(value)
        if not top <= row < canvas.shape[0] - 4:
            continue
        heavy = round(value / step) % heavy_every == 0
        cv2.line(canvas, (0, row), (tick if heavy else tick * 2 // 3, row),
                 (150, 148, 146) if heavy else (86, 84, 82), 1, cv2.LINE_AA)
        if heavy:
            cv2.putText(canvas, f"{value / step * label_step:g}{unit}",
                        (tick + 4, row + 5), cv2.FONT_HERSHEY_SIMPLEX, font,
                        (150, 148, 146), 1, cv2.LINE_AA)
    if datum is None:
        return
    row = camera.row_of(datum)
    if 0 <= row < canvas.shape[0]:
        for x in range(0, canvas.shape[1], 14):
            cv2.line(canvas, (x, row), (min(x + 7, canvas.shape[1]), row),
                     (108, 168, 128), 1, cv2.LINE_AA)


class TextPainter:
    """PIL 画中文标签；cv2.putText 不支持非 ASCII。

    U+2212 MINUS SIGN 与 U+00B2 上标 2 在 Hiragino Sans GB 里缺字形，会渲染成豆腐块，
    所以标签里的负号一律用 ASCII 连字符、平方一律写 `m/s/s`。plot 那侧挑字体时踩的
    是同一个坑。
    """

    FONT = "/System/Library/Fonts/Hiragino Sans GB.ttc"
    SIZES = {"title": 23, "small": 18}

    def __init__(self, path=None, sizes=None):
        from pathlib import Path

        from PIL import ImageFont

        self.path = path or self.FONT
        self.available = Path(self.path).is_file()
        self.sizes = dict(self.SIZES, **(sizes or {}))
        self._cache = {}
        if self.available:
            for kind, size in self.sizes.items():
                self._cache[size] = ImageFont.truetype(self.path, size)
            self.kinds = {k: self._cache[v] for k, v in self.sizes.items()}

    def _font(self, size):
        from PIL import ImageFont

        if size not in self._cache:
            self._cache[size] = ImageFont.truetype(self.path, size)
        return self._cache[size]

    def fitted(self, text, kind, max_width):
        """能放进 `max_width` 的最大字号。

        标签有时是用户给的自由文本，可能比 232 px 的格子宽——裁掉右半截会让
        "修正后(质心=抛物线)"变成"修正后(质心="，读者不知道被截了什么。缩字号是唯一
        不丢信息的办法；缩到 12 px 还放不下就只能裁，但那是极端情况。
        """
        for size in range(self.sizes[kind], 11, -1):
            font = self._font(size)
            if font.getbbox(text)[2] <= max_width:
                return font
        return self._font(12)

    def paint(self, canvas, items):
        """items: [(x, y, 文本, BGR, 'title'|'small')]，原地改写 canvas。"""
        if not self.available:
            for x, y, text, colour, kind in items:
                cv2.putText(canvas, text.encode("ascii", "replace").decode(),
                            (x, y + 18), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6 if kind == "title" else 0.45, colour, 1,
                            cv2.LINE_AA)
            return canvas
        from PIL import Image, ImageDraw

        image = Image.fromarray(canvas[:, :, ::-1])
        draw = ImageDraw.Draw(image)
        for x, y, text, colour, kind in items:
            font = self.fitted(text, kind, canvas.shape[1] - x - 6)
            draw.text((x, y), text, font=font,
                      fill=(colour[2], colour[1], colour[0]))
        canvas[:, :] = np.asarray(image)[:, :, ::-1]
        return canvas


def banner(width, painter, text, height=38, colour=(232, 232, 232)):
    """顶部一条横幅。"""
    strip = np.full((height, width, 3), (18, 17, 16), dtype=np.uint8)
    painter.paint(strip, [(12, 8, text, colour, "small")])
    return strip


def encode(source, path, fps, crf=20, keep=False):
    """H.264 编码。`source` 是 PNG 序列目录或单个视频文件。

    `-pix_fmt yuv420p` 不是可选的：默认的 yuv444p 在 QuickTime 和多数浏览器里放不出
    来。`-vf pad` 把宽高补到偶数——yuv420p 的色度 2×2 下采样，奇数尺寸直接编码失败。
    """
    if source.is_dir():
        args = ["-framerate", str(fps), "-i", str(source / "%04d.png")]
    else:
        args = ["-i", str(source)]
    command = (["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + args
               + ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                  "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
                  "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)])
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        print(result.stderr[-1500:])
        raise RuntimeError(f"ffmpeg 失败，保留中间产物 {source}")
    if not keep and source.is_file():
        source.unlink(missing_ok=True)
    elif not keep:
        for png in source.glob("*.png"):
            png.unlink()
        source.rmdir()
    return path
