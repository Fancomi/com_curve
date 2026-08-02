"""跳水线的统一入口：`python -m python.dive STEP [选项…]`

步骤（可逗号分隔按序执行）：

    extract    frames.bin.br -> outputs/dive_<clip>/curve.{csv,json,npz}
    plot       CSV           -> outputs/dive_<clip>/{height,trajectory,scale}.png
    render     frames.bin.br -> outputs/dive_<clip>/render.mp4（正交四格）
    overlay    frames + 源视频 -> outputs/dive_<clip>/overlay.mp4（叠加）

`extract` 会把源数据路径写进 JSON，之后 `plot/render/overlay` 只给 `--dir` 就够了。
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from ..common import paths as P
from ..common import plotstyle as PS
from ..common import raster as R
from . import align as A
from . import curves as Cv
from . import plots, scenes

STEPS = ("extract", "plot", "render", "overlay")


def _open(args, verbose=True):
    """按参数或已有输出目录打开一段 clip。返回 (clip, 路径字典)。

    `--frames` 与 `--dir` 都没给时退回 outputs 下最新那个 `dive_*`——先 extract 再
    render/overlay 是常规用法，不该要求把源路径再敲一遍。
    """
    frames, metrics = args.frames, args.metrics
    skin, scale = args.skin_obj, args.scale
    out = None
    if args.dir is not None or frames is None:
        out = P.latest_dive_dir(args.dir)
        record = P.read_json(out / "curve.json")
        source = record["source"]
        frames = frames or Path(source["frames"])
        metrics = metrics or Path(source["metrics"])
        skin = skin or Path(source["skin_obj"])
        scale = scale or record["alignment"]["scale_mode"]
    frames = Path(frames).expanduser().resolve()
    if not frames.is_file():
        raise SystemExit(f"frames 不存在：{frames}")
    metrics = P.resolve_metrics(frames, metrics)
    skin = Path(skin or P.DIVE_SKIN_OBJ).expanduser().resolve()
    if not skin.is_file():
        raise SystemExit(f"skin OBJ 不存在：{skin}")

    clip_id = P.clip_id(metrics)
    out = out or P.dive_dir(clip_id)
    out.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"  frames  {frames}")
        print(f"  metrics {metrics}")
        print(f"  out     {P.display(out)}")

    clip = A.Clip(frames, skin, P.read_json(metrics), scale or "metrics",
                  args.fps, args.trim, args.pelvis_count, verbose)
    if clip.rest.shape[0] != clip.header["skin_n"]:
        raise SystemExit(f"skin 顶点数不符：obj={clip.rest.shape[0]} "
                         f"bin={clip.header['skin_n']}")
    return clip, {"frames": frames, "metrics": metrics, "skin_obj": skin,
                  "out": out, "clip": clip_id}


def step_extract(args):
    clip, where = _open(args)
    print(f"  DIVF v{clip.header['version']}  {clip.header['n']} 帧  "
          f"skin={clip.header['skin_n']}  闭合={clip.topology()['closed_manifold']}"
          f"  fps={clip.fps:g}")
    check = None
    if args.check:
        check = clip.reprojection_error()
        print(f"  重投影：{check['count']} 顶点  均值 {check['mean_px']:.6f} px  "
              f"最大 {check['max_px']:.6f} px")

    reference = Cv.metrics_reference(clip.metrics, clip.count)
    out = where["out"]
    print(f"  wrote {P.display(Cv.write_csv(out / 'curve.csv', clip, reference))}")
    meta = Cv.build_meta(clip, {k: where[k] for k in
                                ("frames", "metrics", "skin_obj")}, check)
    print(f"  wrote {P.display(P.write_json(out / 'curve.json', meta))}")
    np.savez_compressed(out / "curve.npz", frames=clip.frames, valid=clip.valid,
                        status=clip.status, root=clip.root, mesh=clip.mesh,
                        volume=clip.volume, fps=np.array(clip.fps),
                        scale=np.array(clip.scale))
    print(f"  wrote {P.display(out / 'curve.npz')}")

    print(f"  可用 mesh 帧 {int((clip.valid == 1).sum())}/{clip.count}，"
          f"体高 {clip.body_height():.2f} m")
    for key, block in clip.fit_flight()["per_curve"].items():
        print(f"    {key:5s} 腾空 g={block['gravity_trimmed_mps2']:+.2f} m/s²  "
              f"顶点 {block['apex_m']:.2f} m  上升 {block['rise_m']:.2f} m  "
              f"残差 {block['fit_rms_trimmed_m']:.3f} m")
    offset = clip.mesh - clip.root
    good = clip.valid == 1
    print(f"    mesh-root ΔY 均值 {np.abs(offset[good, 1]).mean() * 100:.1f} cm，"
          f"最大 {np.abs(offset[good, 1]).max() * 100:.1f} cm")


def step_plot(args):
    from ..common import table as T

    out = P.latest_dive_dir(args.dir)
    data = T.read_table(out / "curve.csv")
    meta = P.read_json(out / "curve.json")
    cjk = PS.setup()
    for path in (plots.height_figure(data, meta, out / "height.png", cjk),
                 plots.trajectory_figure(data, meta, out / "trajectory.png", cjk),
                 plots.scale_figure(data, meta, out / "scale.png", cjk)):
        print(f"  wrote {P.display(path)}")


def step_render(args):
    """正交四格：正视/侧视 × 骨盆代理/Mesh 重心。"""
    import cv2

    from ..common import grid as Grid

    clip, where = _open(args)
    out = where["out"]
    frames_dir = out / "frames"
    if frames_dir.is_dir():
        for stale in frames_dir.glob("*.png"):
            stale.unlink()
    frames_dir.mkdir(parents=True, exist_ok=True)

    total = min(args.max_frames, clip.count) if args.max_frames else clip.count
    cameras = scenes.cameras(clip, args.height)
    painter = R.TextPainter()
    analysis = clip.fit_flight()
    block = (analysis.get("per_curve") or {}).get("mesh") or {}
    events = clip.events()
    width = sum(camera.width for camera in cameras.values()) * len(A.MODES)
    print(f"  {'，'.join(f'{scenes.VIEW_CN[k]} {c.width}px' for k, c in cameras.items())}"
          f"；成片 {width}×{args.height + 40}")

    # mesh 格里额外淡画 root（虚线），让两者的差在同一格里可见；root 格不叠 mesh，
    # 避免两条实线糊在一起分不出主次。副标题按视角变化，把"正视与源视频投影一致 /
    # 侧视只有体厚"写进每一格——这是这条线最容易被误读的地方。
    specs = [Grid.panel(scenes.MODE_CN[key],
                        lambda view, k=key: (f"{scenes.MODE_SUB[k]} · "
                                             f"{scenes.VIEW_SUB[view]}"),
                        scenes.MARKER[key], getattr(clip, key),
                        aux=clip.root if key == "mesh" else None,
                        aux_colour=scenes.MARKER["root"],
                        value=lambda y: (f"Y {y:+.2f} m" if np.isfinite(y)
                                         else "无 mesh"))
             for key in A.MODES]
    scale = {"unit": "m", "step": 1.0, "top": 84, "datum": 0.0, "tick": 22}

    for index in range(total):
        placed = clip.meshes[index]
        bases = {}
        for key, camera in cameras.items():
            canvas = np.full((camera.height, camera.width, 3), R.BACKGROUND,
                             dtype=np.uint8)
            if placed is not None:
                R.draw_mesh(canvas, camera, placed, clip.faces, scenes.BODY_TINT)
            bases[(0, key)] = canvas
        text = (f"跳水重心 · 单目 HSMR   f{index}  t={index / clip.fps:.2f}s"
                f"   尺度 {clip.scale_mode} {clip.scale:.5f} m/px")
        if block:
            text += (f"   腾空 g={block.get('gravity_trimmed_mps2', 0):+.2f} m/s/s"
                     f"  顶点 {block.get('apex_m', 0):.2f} m")
        frame = Grid.compose(bases, specs, scenes.views(), cameras, index, painter,
                             text, scale, trail_span=scenes.TRAIL_SPAN,
                             banner_height=40,
                             extra=scenes.event_badge(width, index, events))
        cv2.imwrite(str(frames_dir / f"{index + 1:04d}.png"), frame)
        if (index + 1) % 100 == 0 or index + 1 == total:
            print(f"    {index + 1}/{total}")

    video = R.encode(frames_dir, out / "render.mp4", clip.fps, args.crf,
                     keep=args.keep_frames)
    print(f"  wrote {P.display(video)} ({video.stat().st_size / 1e6:.1f} MB)")


def step_overlay(args):
    """叠回源视频：半透明人体 + 质心与轨迹 + 米制标尺 + 右侧曲线板。"""
    import cv2

    clip, where = _open(args)
    out = where["out"]
    video = P.find_video(where["metrics"], where["clip"], args.video)
    print(f"  video   {video}")

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SystemExit(f"无法打开 {video}")
    source_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (source_w, source_h) != (int(A.SOURCE_W), int(A.SOURCE_H)):
        print(f"  注意：底图 {source_w}x{source_h} 与标定假设的 "
              f"{int(A.SOURCE_W)}x{int(A.SOURCE_H)} 不同，投影会错位")

    camera = R.PixelCamera(clip.scale, source_w, source_h)
    good = clip.valid == 1
    low = float(np.nanmin(clip.mesh[good, 1])) - 1.0
    high = float(np.nanmax(clip.mesh[good, 1])) + 1.0
    velocity = Cv.speeds(clip)
    curves = {"frames": clip.frames, "valid": clip.valid}
    for key in A.MODES:
        curves[f"{key}_y"] = getattr(clip, key)[:, 1]
        curves[f"{key}_vy"] = velocity[key][:, 1]
    meta = {"fps": clip.fps, "events": clip.events(), "flight": clip.fit_flight(),
            "alignment": {"scale_mode": clip.scale_mode,
                          "scale_m_per_px": clip.scale,
                          "body_height_m": clip.body_height()}}

    panel_width = max(args.panel_width, 0)
    out_w = source_w + panel_width + (source_w + panel_width) % 2
    raw = out / "overlay.raw.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"),
                             clip.fps, (out_w, source_h))
    if not writer.isOpened():
        raise SystemExit("VideoWriter 打开失败")
    total = min(args.max_frames, clip.count) if args.max_frames else clip.count
    print(f"  成片 {out_w}x{source_h} @ {clip.fps:g} fps，"
          f"可用 {int(good.sum())}/{clip.count}")

    painter = R.TextPainter()
    panel = (np.zeros((source_h, panel_width, 3), np.uint8) if panel_width
             else None)
    rim = R.rim_edges(clip.faces)

    for index in range(total):
        ok, frame = capture.read()
        if not ok:
            frame = np.zeros((source_h, source_w, 3), np.uint8)
        elif frame.shape[:2] != (source_h, source_w):
            frame = cv2.resize(frame, (source_w, source_h))

        # 标尺零点是画面中心那条水平线（固定深度平面的 Y=0），不是水面。
        R.draw_scale(frame, camera, unit="m", step=1.0, top=0, datum=0.0,
                     heavy_every=2, tick=34, font=0.5)
        placed = clip.meshes[index]
        if placed is not None and not args.no_mesh:
            R.draw_mesh(frame, camera, placed, clip.faces, scenes.OVERLAY_TINT,
                        alpha=args.alpha, silhouette=rim,
                        rim_colour=scenes.SILHOUETTE)
        for key, thickness in (("root", 1), ("mesh", 3)):
            R.draw_trail(frame, camera, getattr(clip, key), index,
                         scenes.MARKER[key], span=args.recent,
                         thickness=thickness, background=None,
                         faint=R.fade(scenes.MARKER[key], 0.42))
        R.draw_marker(frame, camera, clip.root[index], scenes.MARKER["root"],
                      hollow=True, style="ring")
        R.draw_marker(frame, camera, clip.mesh[index], scenes.MARKER["mesh"],
                      style="ring")
        scenes.draw_hud(frame, painter, curves, meta, index, bool(good[index]))

        if panel is not None:
            scenes.draw_panel(panel, curves, meta, index)
            frame = np.concatenate([frame, panel], axis=1)
        if frame.shape[1] != out_w:
            frame = cv2.copyMakeBorder(frame, 0, 0, 0, out_w - frame.shape[1],
                                       cv2.BORDER_CONSTANT,
                                       value=scenes.PANEL_BG)
        writer.write(frame)
        if index % 100 == 0:
            print(f"    {index}/{total}")

    writer.release()
    capture.release()
    video_out = R.encode(raw, out / "overlay.mp4", clip.fps, args.crf)
    print(f"  wrote {P.display(video_out)} "
          f"({video_out.stat().st_size / 1e6:.1f} MB)")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m python.dive", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("steps", help="逗号分隔：" + ",".join(STEPS))
    parser.add_argument("--frames", type=Path, default=None,
                        help="*_replay_frames.bin[.br]")
    parser.add_argument("--metrics", type=Path, default=None,
                        help="配套 *_replay_metrics.json（默认按命名推断）")
    parser.add_argument("--skin-obj", type=Path, default=None,
                        help="HSMR skin 拓扑 OBJ")
    parser.add_argument("--dir", type=Path, default=None,
                        help="已有的输出目录（从它的 curve.json 读回源路径）")
    parser.add_argument("--scale", choices=A.SCALE_CHOICES, default=None,
                        help="米每像素的来源（默认 metrics，见 dive.align 文档）")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--trim", type=int, default=8,
                        help="腾空段重力拟合两端切除帧数")
    parser.add_argument("--pelvis-count", type=int, default=48,
                        help="骨盆代理取多少个顶点")
    parser.add_argument("--check", action="store_true",
                        help="逐顶点验证 global -> 2D 重投影（约 1 s）")
    # render / overlay
    parser.add_argument("--video", type=Path, default=None,
                        help="overlay 的底图视频（默认原始机位 <clip>.mp4）")
    parser.add_argument("--height", type=int, default=680,
                        help="render 单格像素高度")
    parser.add_argument("--panel-width", type=int, default=520,
                        help="overlay 右侧曲线板宽度；0 表示不要")
    parser.add_argument("--recent", type=int, default=90,
                        help="overlay 轨迹加亮的帧数")
    parser.add_argument("--no-mesh", action="store_true",
                        help="overlay 不画人体，只画质心与轨迹")
    parser.add_argument("--alpha", type=float, default=scenes.MESH_ALPHA)
    parser.add_argument("--max-frames", type=int, default=0,
                        help="只渲染前若干帧（调试用）")
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--keep-frames", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    actions = {"extract": step_extract, "plot": step_plot,
               "render": step_render, "overlay": step_overlay}
    chosen = [s.strip() for s in args.steps.split(",") if s.strip()]
    unknown = [s for s in chosen if s not in actions]
    if unknown:
        raise SystemExit(f"未知步骤 {unknown}，可选：{', '.join(STEPS)}")
    for step in chosen:
        print(f"[{step}]")
        actions[step](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
