#!/usr/bin/env python3
"""Profile the dive pipeline without changing production code.

The script drives the same extract/plot/render/overlay entry points as dive.sh,
then times their hot functions. Outputs are placed in a temporary directory by
default and the report is JSON so it can be inspected or compared later.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.common import grid as Grid
from python.common import paths as P
from python.common import raster as R
from python.common import table as T
from python.dive import __main__ as dive_main
from python.dive import align as A
from python.dive import curves as Cv
from python.dive import divf
from python.dive import plots
from python.dive import scenes


DEFAULT_SKIN = P.DIVE_SKIN_OBJ
DEFAULT_FRAMES = [
    Path("/Users/penghaotian/Downloads/DATAS/DIVING/data_daily/20260731/"
         "20260731-192305_replay_frames.bin.br"),
    Path("/Users/penghaotian/Downloads/DATAS/DIVING/data_daily/20260801/"
         "20260801-184501_replay_frames.bin.br"),
]


class Probe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.samples = defaultdict(list)

    def call(self, label, fn, *args, **kwargs):
        start = time.perf_counter_ns()
        try:
            return fn(*args, **kwargs)
        finally:
            self.samples[label].append(time.perf_counter_ns() - start)

    def stats(self):
        result = {}
        for label, values in sorted(self.samples.items()):
            ms = np.asarray(values, dtype=np.float64) / 1e6
            result[label] = {
                "calls": int(ms.size),
                "total_ms": float(ms.sum()),
                "mean_ms": float(ms.mean()),
                "p50_ms": float(np.percentile(ms, 50)),
                "p95_ms": float(np.percentile(ms, 95)),
                "p99_ms": float(np.percentile(ms, 99)),
                "max_ms": float(ms.max()),
            }
        return result


class PatchSet:
    def __init__(self, probe):
        self.probe = probe
        self.originals = []

    def replace(self, owner, name, label=None):
        original = getattr(owner, name)
        self.originals.append((owner, name, original))
        label = label or name

        def wrapped(*args, **kwargs):
            return self.probe.call(label, original, *args, **kwargs)

        setattr(owner, name, wrapped)

    def restore(self):
        for owner, name, original in reversed(self.originals):
            setattr(owner, name, original)
        self.originals.clear()


class CaptureProxy:
    def __init__(self, probe, original):
        self._probe = probe
        self._inner = original

    def read(self):
        return self._probe.call("overlay.video_read", self._inner.read)

    def get(self, *args):
        return self._inner.get(*args)

    def isOpened(self):
        return self._inner.isOpened()

    def release(self):
        return self._inner.release()


class WriterProxy:
    def __init__(self, probe, original):
        self._probe = probe
        self._inner = original

    def write(self, *args):
        return self._probe.call("overlay.video_write", self._inner.write, *args)

    def isOpened(self):
        return self._inner.isOpened()

    def release(self):
        return self._inner.release()


def install_patches(probe):
    """Patch only module-level references used by the existing pipeline."""
    patches = PatchSet(probe)

    for owner, name, label in (
        (divf, "read", "io.divf_read_decompress"),
        (divf, "header", "decode.header"),
        (divf, "record", "decode.record"),
        (divf, "load_obj", "mesh.load_obj"),
        (A, "project_source", "frame.project_source"),
        (A, "place_on_plane", "mesh.place_on_plane"),
        (A, "hsmr_scale", "scale.hsmr"),
        (A, "gravity_scale", "scale.gravity"),
        (A.G, "volume_centroid", "mesh.volume_centroid"),
        (A.G, "edge_owners", "topology.edge_owners"),
        (A.G, "is_closed", "topology.is_closed"),
        (A.Clip, "fit_flight", "analysis.fit_flight"),
        (A.Clip, "events", "analysis.events"),
        (Grid, "frame_ranges", "render.frame_ranges"),
        (Grid, "draw_cell", "render.draw_cell"),
        (Grid, "compose", "render.compose"),
        (scenes, "cameras", "render.cameras"),
        (scenes, "draw_panel", "overlay.panel"),
        (scenes, "draw_hud", "overlay.hud"),
        (R, "draw_mesh", "render.draw_mesh"),
        (R, "draw_trail", "render.draw_trail"),
        (R, "draw_marker", "render.draw_marker"),
        (R, "draw_scale", "overlay.draw_scale"),
        (R, "rim_edges", "render.rim_edges"),
        (R, "encode", "io.ffmpeg_encode"),
        (Cv, "write_csv", "io.write_csv"),
        (P, "write_json", "io.write_json"),
        (T, "read_table", "io.read_table"),
        (dive_main.np, "savez_compressed", "io.npz_compressed"),
        (plots, "height_figure", "plot.height_figure"),
        (plots, "trajectory_figure", "plot.trajectory_figure"),
        (plots, "scale_figure", "plot.scale_figure"),
        (dive_main, "_open", "pipeline.open_clip"),
    ):
        patches.replace(owner, name, label)

    # These are constructors, so use small proxies rather than timing only setup.
    import cv2

    patches.replace(cv2, "imwrite", "io.png_write")
    original_capture = cv2.VideoCapture
    original_writer = cv2.VideoWriter
    patches.originals.append((cv2, "VideoCapture", original_capture))
    patches.originals.append((cv2, "VideoWriter", original_writer))
    cv2.VideoCapture = lambda *args, **kwargs: CaptureProxy(
        probe, original_capture(*args, **kwargs))
    cv2.VideoWriter = lambda *args, **kwargs: WriterProxy(
        probe, original_writer(*args, **kwargs))

    return patches


def args_for(frames, metrics, skin, out, video, step):
    return SimpleNamespace(
        frames=frames,
        metrics=metrics,
        skin_obj=skin,
        scale="metrics",
        fps=None,
        trim=8,
        verbose=False,
        dir=out if step != "extract" else None,
        check=False,
        height=540,
        crf=23,
        keep_frames=False,
        max_frames=None,
        video=video,
        panel_width=420,
        alpha=0.28,
        recent=45,
        no_mesh=False,
    )


def run_step(probe, step, args):
    probe.reset()
    start = time.perf_counter_ns()
    if step == "extract":
        dive_main.step_extract(args)
    elif step == "plot":
        dive_main.step_plot(args)
    elif step == "render":
        dive_main.step_render(args)
    elif step == "overlay":
        dive_main.step_overlay(args)
    else:
        raise ValueError(step)
    elapsed = (time.perf_counter_ns() - start) / 1e6
    result = {"elapsed_ms": elapsed, "timings": probe.stats()}
    result["timings"]["__step_total__"] = {
        "calls": 1,
        "total_ms": elapsed,
        "mean_ms": elapsed,
        "p50_ms": elapsed,
        "p95_ms": elapsed,
        "p99_ms": elapsed,
        "max_ms": elapsed,
    }
    return result


def microbench(frames, skin, repeats):
    raw = divf.read(frames)
    header = divf.header(raw)
    rest, faces = divf.load_obj(skin)
    first = None
    for index in range(header["n"]):
        record = divf.record(raw, header, index)
        if record["usable"]:
            first = record
            break
    if first is None:
        raise ValueError(f"no usable frames in {frames}")
    local = first["skin"]
    placed = A.place_on_plane(local, first, 0.01, 0.01)

    def repeat(fn, value):
        samples = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            fn(value, faces)
            samples.append((time.perf_counter_ns() - start) / 1e6)
        return {
            "calls": repeats,
            "mean_ms": float(np.mean(samples)),
            "p50_ms": float(np.percentile(samples, 50)),
            "p95_ms": float(np.percentile(samples, 95)),
            "min_ms": float(np.min(samples)),
            "max_ms": float(np.max(samples)),
        }

    return {
        "vertices": int(local.shape[0]),
        "faces": int(faces.shape[0]),
        "local_volume_centroid": repeat(A.G.volume_centroid, local),
        "placed_volume_centroid": repeat(A.G.volume_centroid, placed),
    }


def summarize(result):
    lines = []
    lines.append("\nPerformance summary")
    lines.append("clip | step | elapsed_ms | top measured functions")
    lines.append("--- | --- | ---: | ---")
    for clip_name, clip in result["clips"].items():
        for step, info in clip["steps"].items():
            rows = [
                (name, data["total_ms"])
                for name, data in info["timings"].items()
                if name != "__step_total__"
            ]
            rows.sort(key=lambda item: item[1], reverse=True)
            top = ", ".join(f"{name}={value:.1f}" for name, value in rows[:4])
            lines.append(f"{clip_name} | {step} | {info['elapsed_ms']:.1f} | {top}")
    lines.append("")
    lines.append("Mesh centroid call breakdown")
    lines.append("clip | step | calls | total_ms | mean_us | p50_us | p95_us")
    lines.append("--- | --- | ---: | ---: | ---: | ---: | ---:")
    for clip_name, clip in result["clips"].items():
        for step, info in clip["steps"].items():
            data = info["timings"].get("mesh.volume_centroid")
            if data:
                lines.append(
                    f"{clip_name} | {step} | {data['calls']} | "
                    f"{data['total_ms']:.1f} | {data['mean_ms'] * 1000:.1f} | "
                    f"{data['p50_ms'] * 1000:.1f} | {data['p95_ms'] * 1000:.1f}")
    return "\n".join(lines)


def profile_clip(frames, skin, video, root, probe):
    frames = Path(frames).expanduser().resolve()
    metrics = P.resolve_metrics(frames)
    clip_name = P.clip_id(metrics)
    out = root / clip_name
    out.mkdir(parents=True, exist_ok=True)

    # Keep the production path unchanged but redirect its output directory.
    original_dive_dir = P.dive_dir
    original_latest = P.latest_dive_dir
    P.dive_dir = lambda _clip: out
    P.latest_dive_dir = lambda explicit=None: Path(explicit or out)
    try:
        steps = {}
        for step in ("extract", "plot", "render", "overlay"):
            step_args = args_for(frames, metrics, skin, out, video, step)
            steps[step] = run_step(probe, step, step_args)
        return {
            "frames": str(frames),
            "metrics": str(metrics),
            "video": str(video),
            "frame_count": int(divf.header(divf.read(frames))["n"]),
            "steps": steps,
            "microbench": microbench(frames, skin, repeats=40),
        }
    finally:
        P.dive_dir = original_dive_dir
        P.latest_dive_dir = original_latest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", nargs="*", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--skin", type=Path, default=DEFAULT_SKIN)
    parser.add_argument("--video", nargs="*", type=Path, default=None)
    parser.add_argument("--report", type=Path,
                        default=Path("outputs/dive_performance.json"))
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args()

    if len(args.frames) not in (1, 2):
        parser.error("pass one or two --frames paths")
    videos = args.video or [None] * len(args.frames)
    if len(videos) != len(args.frames):
        parser.error("--video must have one path per --frames path")
    if any(not path.is_file() for path in args.frames):
        parser.error("one or more --frames files do not exist")
    if not args.skin.is_file():
        parser.error(f"skin OBJ does not exist: {args.skin}")
    if any(path is not None and not path.is_file() for path in videos):
        parser.error("one or more --video files do not exist")

    work = Path(tempfile.mkdtemp(prefix="dive_profile_"))
    probe = Probe()
    patches = install_patches(probe)
    result = {
        "script": str(Path(__file__).resolve()),
        "python": __import__("sys").executable,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "skin": str(args.skin.resolve()),
        "clips": {},
        "work_dir": str(work),
    }
    try:
        for frames, video in zip(args.frames, videos):
            if video is None:
                replay_stem = frames.name.replace("_replay_frames.bin.br", "")
                candidates = [frames.with_name(replay_stem + ".mp4")]
                if "-" in replay_stem:
                    date, clock = replay_stem.split("-", 1)
                    candidates.append(frames.with_name(f"{date}_{clock}.mp4"))
                video = next((candidate for candidate in candidates
                              if candidate.is_file()), None)
            if video is None:
                raise SystemExit(f"no source video supplied for {frames}")
            name = P.clip_id(P.resolve_metrics(frames))
            print(f"\nProfiling {name}")
            result["clips"][name] = profile_clip(
                frames, args.skin.resolve(), video.resolve(), work, probe)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        print(summarize(result))
        print(f"\nJSON report: {args.report.resolve()}")
    finally:
        patches.restore()
        if not args.keep_work:
            shutil.rmtree(work, ignore_errors=True)
        else:
            print(f"Temporary outputs kept at: {work}")


if __name__ == "__main__":
    main()
