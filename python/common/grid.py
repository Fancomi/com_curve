"""「视角 × 面板」的四格拼图。两条 pipeline 的对照视频共用这一份。

一个**面板**是一列要画的东西：标签、颜色、一条主曲线、可选的副曲线（虚线）、可选的
体段质心圆，以及它从哪张底图上画（`source`）。同一场景比两种口径时各面板共用底图；
比两个文件时各用自己的。把这件事收进 `source` 而不是分两套 compose，是因为除底图外
的一切（刻度、轨迹、标记、标签、横幅）逐像素相同。

四格排成**一横排**而不是 2×2：这些场景垂直跨 6 m、水平只有 2~5 m，2×2 会得到又高又
窄的竖片，而且要比较的两格上下相隔 600 px，眼睛得跨过一整格。横排下四格共用同一条
垂直刻度，那条重心高度线跨整幅画面处在同一行——横着扫一眼就能看出两条曲线差多少。
"""
import cv2
import numpy as np

from . import raster as R

CELL_BORDER = (64, 62, 60)
LABEL_X = 10
LABEL_ROWS = (6, 33, 55, 77)     # 标题 / 副标题 / 主读数 / 副读数
SUB_COLOUR = (170, 168, 166)
VALUE_COLOUR = (226, 226, 226)


def panel(label, sub, colour, track, source=0, aux=None, aux_colour=None,
          aux_label=None, segments=None, weights=None, full_scale=None,
          value=None):
    """一个面板的描述。

    `sub` 是副标题，给 `f(view_key) -> str` 时按视角变化——dive 那条线要在副标题里
    说明"正视与源视频投影一致 / 侧视只有体厚"，那是跨视角不同、跨面板相同的信息。
    `value` 是主读数的格式化函数 `f(y) -> str`。
    """
    return {"label": label, "sub": sub, "colour": colour, "track": track,
            "source": source, "aux": aux, "aux_colour": aux_colour,
            "aux_label": aux_label, "segments": segments, "weights": weights,
            "full_scale": full_scale, "value": value}


def draw_cell(base, camera, spec, index, painter, view_key, view_label,
              scale=None, marker_style="disc", trail_span=45):
    """一格：底图 + 刻度 + 副曲线 + 主曲线 + 标签。返回新画布。"""
    canvas = base.copy()
    if scale is not None:
        R.draw_scale(canvas, camera, **scale)
    colour = spec["colour"]
    if spec["segments"] is not None:
        R.draw_discs(canvas, camera, spec["segments"][index], spec["weights"],
                     colour, spec["full_scale"])
    # 副轨迹先画：虚线压在实线下面，实线才是这一格的主角。
    if spec["aux"] is not None:
        R.draw_trail(canvas, camera, spec["aux"], index, spec["aux_colour"],
                     span=trail_span, dashed=True)
        R.draw_marker(canvas, camera, spec["aux"][index], spec["aux_colour"],
                      hollow=True, style=marker_style)
    R.draw_trail(canvas, camera, spec["track"], index, colour, span=trail_span)
    R.draw_marker(canvas, camera, spec["track"][index], colour,
                  style=marker_style)

    height = spec["track"][index, 1]
    reading = spec["value"](height) if spec["value"] else f"{height:.2f}"
    sub = spec["sub"](view_key) if callable(spec["sub"]) else spec["sub"]
    lines = [(LABEL_X, LABEL_ROWS[0], f"{view_label} · {spec['label']}",
              colour, "title"),
             (LABEL_X, LABEL_ROWS[1], sub, SUB_COLOUR, "small"),
             (LABEL_X, LABEL_ROWS[2], reading, VALUE_COLOUR, "small")]
    if spec["aux"] is not None and spec["aux_label"]:
        aux_y = spec["aux"][index, 1]
        lines.append((LABEL_X, LABEL_ROWS[3],
                      f"- - {spec['aux_label']} "
                      + (spec["value"](aux_y) if spec["value"] else f"{aux_y:.2f}"),
                      spec["aux_colour"], "small"))
    painter.paint(canvas, lines)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, canvas.shape[0] - 1),
                  CELL_BORDER, 1)
    return canvas


def compose(bases, panels, views, cameras, index, painter, banner_text,
            scale=None, marker_style="disc", trail_span=45, banner_height=38,
            extra=None):
    """整帧：视角外层、面板内层，横向拼接，顶上一条横幅。

    `bases` 的键是 (source, 视角名)。`extra` 可选，画在横幅右侧（事件标记之类）。
    """
    cells = []
    for view_key, view_label in views:
        camera = cameras[view_key]
        for spec in panels:
            cells.append(draw_cell(bases[(spec["source"], view_key)], camera,
                                   spec, index, painter, view_key, view_label,
                                   scale, marker_style, trail_span))
    grid = np.hstack(cells)
    strip = R.banner(grid.shape[1], painter, banner_text, banner_height)
    if extra:
        painter.paint(strip, extra)
    frame = np.vstack([strip, grid])
    # yuv420p 的色度 2×2 下采样要求两个维度都是偶数。
    pad_h, pad_w = frame.shape[0] % 2, frame.shape[1] % 2
    if pad_h or pad_w:
        frame = cv2.copyMakeBorder(frame, 0, pad_h, 0, pad_w,
                                   cv2.BORDER_CONSTANT, value=R.BACKGROUND)
    return frame


def frame_ranges(clouds, tracks, pad=0.0, floor=None, headroom=0.0, min_span=0.6):
    """四格共用的垂直范围 + 每个轴的水平范围。

    垂直范围必须共用，否则各格的高度线不在同一行，"这条比那条高 5 cm"就读不出来。
    水平范围各视角按自己的跨度算：一个视角要装下 4 m 的横向位移，另一个只有半米的
    体厚，共用宽度会让后者缩成一条。

    `clouds` 是若干 (N, 3) 点云（人体包围盒的角点即可），`tracks` 是若干曲线，
    nan 会被跳过。`floor` 给了就把垂直下界钉到它（地面），`headroom` 在上方留余量
    给标记环与轨迹。`pad` 是按跨度比例的两侧留白。
    """
    stack = [np.atleast_2d(c) for c in clouds if c is not None and len(c)]
    low = np.min([c.min(0) for c in stack], axis=0)
    high = np.max([c.max(0) for c in stack], axis=0)
    for track in tracks:
        good = np.isfinite(track).all(axis=1)
        if good.any():
            low = np.minimum(low, track[good].min(0))
            high = np.maximum(high, track[good].max(0))
    span = np.maximum(high - low, min_span)
    low, high = low - pad * span, high + pad * span
    y_low = low[1] if floor is None else min(floor, low[1])
    return ([(float(low[k]), float(high[k])) for k in range(3)],
            (float(y_low), float(high[1] + headroom)))
