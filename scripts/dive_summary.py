#!/usr/bin/env python3
"""汇总 outputs/dive_*/curve.json：mesh/gb 的 g 拟合、mesh→gb 差、gb 段长偏差。

批量跑完之后回答一个问题：**gb 口径跨 clip 稳不稳。** 标注是同一份，所以段长占身高
的比例应当基本一致（差异只来自姿态与 HSMR 的体型估计）；mesh→gb 差若在各 clip 上
量级一致，那 `gb ≈ mesh` 就不是单个 clip 的巧合。

    python3 scripts/dive_summary.py
    python3 scripts/dive_summary.py --csv outputs/dive_summary.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.common import paths as P            # noqa: E402


def read_curve(directory):
    """一个 clip 的 (meta, 逐帧列)。缺 csv/json 的目录跳过。"""
    meta_path, csv_path = directory / "curve.json", directory / "curve.csv"
    if not (meta_path.is_file() and csv_path.is_file()):
        return None, None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    columns = {name: np.array([float(r[name]) if r[name] else np.nan
                               for r in rows])
               for name in rows[0]}
    return meta, columns


def pair_stats(columns, left, right):
    """两条曲线的 ΔY 与三维距离。只统计 valid 帧。"""
    good = columns["valid"] > 0.5
    delta_y = columns[f"{right}_y_m"] - columns[f"{left}_y_m"]
    first = np.stack([columns[f"{left}_{a}_m"] for a in "xyz"], 1)
    second = np.stack([columns[f"{right}_{a}_m"] for a in "xyz"], 1)
    distance = np.linalg.norm(second - first, axis=1)
    mask = good & np.isfinite(delta_y) & np.isfinite(distance)
    if not mask.any():
        return None
    return {"dy_mean_cm": float(delta_y[mask].mean() * 100),
            "dy_std_cm": float(delta_y[mask].std() * 100),
            "dy_max_abs_cm": float(np.abs(delta_y[mask]).max() * 100),
            "dist_mean_cm": float(distance[mask].mean() * 100),
            "dist_max_cm": float(distance[mask].max() * 100)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=None,
                        help="同时写一份逐 clip 的 CSV")
    args = parser.parse_args()

    records = []
    for directory in sorted(P.OUTPUTS.glob("dive_*")):
        if not directory.is_dir():
            continue
        meta, columns = read_curve(directory)
        if meta is None:
            continue
        modes = list((meta.get("modes") or {}).keys())
        per_curve = (meta.get("flight") or {}).get("per_curve") or {}
        records.append({
            "clip": directory.name.replace("dive_", ""),
            "modes": modes,
            "usable": meta.get("usable_frames"),
            "frames": meta["divf_header"]["n"],
            "scale_mode": meta["alignment"]["scale_mode"],
            "scale": meta["alignment"]["scale_m_per_px"],
            "height": meta["alignment"]["body_height_m"],
            "gravity": {k: v.get("gravity_trimmed_mps2")
                        for k, v in per_curve.items()},
            "rms": {k: v.get("fit_rms_trimmed_m") for k, v in per_curve.items()},
            "gb": meta.get("gb"),
            "pairs": ({"mesh_to_gb": pair_stats(columns, "mesh", "gb")}
                      if {"mesh", "gb"} <= set(modes) else {}),
        })

    if not records:
        raise SystemExit(f"{P.display(P.OUTPUTS)} 下没有 dive_*/curve.json")

    with_gb = [r for r in records if "gb" in r["modes"]]
    print(f"{len(records)} 个 clip，其中 {len(with_gb)} 个有 gb 曲线\n")

    header = (f"{'clip':17s} {'可用':>5s} {'尺度':>7s} {'s':>9s} {'体高':>5s} "
              f"{'g_mesh':>7s} {'g_gb':>7s} {'残差gb':>7s}")
    print(header)
    print("-" * len(header))
    for r in records:
        g = r["gravity"]
        cell = lambda value: f"{value:+7.2f}" if value is not None else "      -"
        rms = r["rms"].get("gb")
        print(f"{r['clip']:17s} {r['usable']:5d} {r['scale_mode']:>7s} "
              f"{r['scale']:9.6f} {r['height']:5.2f} "
              f"{cell(g.get('mesh'))} {cell(g.get('gb'))} "
              f"{(f'{rms:7.3f}' if rms is not None else '      -')}")

    print("\nmesh→gb 差（cm，只统计 valid 帧）")
    print(f"{'clip':17s} {'mesh→gb ΔY':>22s}")
    for r in with_gb:
        stat = r["pairs"].get("mesh_to_gb")
        print(f"{r['clip']:17s} "
              + (f"{stat['dy_mean_cm']:+6.2f}±{stat['dy_std_cm']:4.2f} "
                 f"max{stat['dy_max_abs_cm']:5.2f}" if stat else " " * 22))

    # 跨 clip 的一致性：同一份标注，段长比例应当基本一致。
    if with_gb:
        print("\ngb 段长占身高（%）跨 clip 的散布")
        names = [e["name"] for e in with_gb[0]["gb"]["segments"]]
        print(f"{'段':13s} {'均值':>7s} {'σ':>6s} {'极差':>6s} {'国标':>7s} {'差':>7s}")
        for index, name in enumerate(names):
            values = np.array([r["gb"]["segments"][index]["pct_of_stature"]
                               for r in with_gb])
            reference = with_gb[0]["gb"]["segments"][index]["gb_pct_of_stature"]
            print(f"{name:13s} {values.mean():7.2f} {values.std():6.3f} "
                  f"{np.ptp(values):6.3f} {reference:7.2f} "
                  f"{values.mean() - reference:+7.2f}")

        statures = np.array([r["gb"]["stature_m"] for r in with_gb])
        print(f"\n标注身高：{statures.mean() * 100:.2f} ± "
              f"{statures.std() * 100:.2f} cm（极差 {np.ptp(statures) * 100:.2f}）")
        fallbacks = {k for r in with_gb for k in (r["gb"]["fallbacks"] or {})}
        if fallbacks:
            worst = max(v for r in with_gb
                        for v in (r["gb"]["fallbacks"] or {}).values())
            print(f"退化点：{'、'.join(sorted(fallbacks))}，最大 {worst:.1f} mm"
                  f"（标注顶点不在 DIVF 子集内）")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["clip", "usable", "frames", "scale_mode", "scale",
                             "height_m", "g_mesh", "g_gb",
                             "rms_gb_m", "mesh_to_gb_dy_cm", "gb_stature_m"])
            for r in records:
                pair = r["pairs"].get("mesh_to_gb") or {}
                writer.writerow([
                    r["clip"], r["usable"], r["frames"], r["scale_mode"],
                    round(r["scale"], 8), r["height"],
                    r["gravity"].get("mesh"), r["gravity"].get("gb"),
                    r["rms"].get("gb"),
                    round(pair.get("dy_mean_cm", float("nan")), 3),
                    (r["gb"] or {}).get("stature_m")])
        print(f"\nwrote {P.display(args.csv)}")


if __name__ == "__main__":
    main()
