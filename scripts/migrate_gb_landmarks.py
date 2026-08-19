#!/usr/bin/env python3
"""把稀疏子集上标的 gb_landmarks.json 迁到稠密索引。

早先的标注页面渲染的是 HSMR 简化 OBJ（skin 1379 / skel 3271），顶点索引指向那两个
子集。现在页面直接用稠密网格（skin 6890 / skel 247252），索引口径变了。

迁移是**精确查表**，不是最近点近似：子集顶点 i 就是稠密网格的 `mapping[i]`，
两个 mapping 由 HSMR 权重目录提供。实测 20 个点的坐标偏差最大 0.0065 mm
（float32 往返的舍入），所以这一步不引入任何几何误差。

    python3 scripts/migrate_gb_landmarks.py ~/Downloads/gb_landmarks.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.common import paths as P            # noqa: E402

HSMR_DIR = Path("/Users/penghaotian/Documents/GIT/rtsp-h264-stitcher/weights/"
                "diving/hsmr_official")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="稀疏口径的 gb_landmarks.json")
    parser.add_argument("--hsmr", type=Path, default=HSMR_DIR)
    parser.add_argument("--asset", type=Path,
                        default=P.OUTPUTS / "gb_landmark_asset.json",
                        help="稠密 manifest，用来核对顶点数与坐标")
    parser.add_argument("--out", type=Path, default=None,
                        help="默认写到 inputs/gb_landmarks.json")
    args = parser.parse_args()

    old = json.loads(args.source.read_text(encoding="utf-8"))
    if old.get("skin_count") not in (1379, None):
        raise SystemExit(f"这个文件的 skin_count 是 {old.get('skin_count')}，"
                         f"不是 1379 —— 可能已经是稠密口径了")

    mapping = {
        "skin": np.load(args.hsmr / "skin_vertex_mapping.npy"),
        "skel": np.load(args.hsmr / "skel_vertex_mapping.npy"),
    }
    asset = json.loads(args.asset.read_text(encoding="utf-8"))
    dense = {}
    for layer in ("skin", "skel"):
        block = asset[layer]
        raw = (args.asset.parent / block["file"]).read_bytes()
        start = block["offsets"]["positions"]
        dense[layer] = np.frombuffer(
            raw, "<f4", block["count"] * 3, start).reshape(-1, 3)

    out = {}
    worst = 0.0
    print(f"{'id':16s} {'层':5s} {'子集':>6s} -> {'稠密':>8s}   坐标核对")
    for key, entry in old["landmarks"].items():
        layer = entry.get("mesh", "skin")
        index = int(mapping[layer][entry["vertex"]])
        reference = dense[layer][index]
        error = float(np.linalg.norm(reference - np.array(entry["xyz"])))
        worst = max(worst, error)
        out[key] = {**entry, "vertex": index,
                    "xyz": [round(float(v), 5) for v in reference],
                    "was_subset_vertex": entry["vertex"]}
        print(f"{key:16s} {layer:5s} {entry['vertex']:6d} -> {index:8d}   "
              f"Δ = {error * 1000:.4f} mm")

    if worst > 1e-4:
        raise SystemExit(f"坐标偏差 {worst * 1000:.3f} mm 太大，映射对不上")

    payload = {
        "standard": old.get("standard"),
        "note": "vertex 是 SKEL/SMPL 原生索引（skin 6890 / skel 247252），"
                "无抽稀、无中间映射；下游 verts[index] 直接可用",
        "pose": asset["source"]["pose"],
        "axes": asset["axes"],
        "skin_count": asset["skin"]["count"],
        "skel_count": asset["skel"]["count"],
        "migrated_from": {"file": str(args.source),
                          "skin_count": old.get("skin_count"),
                          "skel_count": old.get("skel_count")},
        "complete": len(out) == 20,
        "picked": len(out),
        "total": 20,
        "landmarks": out,
    }
    target = args.out or (P.ROOT / "inputs" / "gb_landmarks.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(f"\n最大坐标偏差 {worst * 1000:.4f} mm（float32 舍入量级）")
    print(f"wrote {P.display(target)}（{len(out)} 个点）")


if __name__ == "__main__":
    main()
