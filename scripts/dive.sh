#!/usr/bin/env bash
# 跳水 DIVF replay 线的统一入口 —— 从 *_replay_frames.bin.br 提取重心曲线。
#
#   ./scripts/dive.sh STEPS [选项…]
#
# STEPS 逗号分隔、按给出顺序执行：
#   extract  frames.bin.br -> outputs/dive_<clip>/curve.{csv,json,npz}
#            两种口径：mesh(体积重心) / gb(国标 15 段·体表骨突)，单位米。
#            gb 要 inputs/gb_landmarks.json，缺了自动跳过。
#   plot     CSV -> height.png / trajectory.png / scale.png
#   render   正交四格视频（无底图）-> outputs/dive_<clip>/render.mp4
#   overlay  叠回源视频 -> outputs/dive_<clip>/overlay.mp4
#
# 常用：
#   ./scripts/dive.sh extract --frames PATH/xxx_replay_frames.bin.br --check
#   ./scripts/dive.sh plot,render,overlay --dir outputs/dive_20260731-192305
#   ./scripts/dive.sh extract --frames F --scale gravity     # 换尺度口径
#   ./scripts/dive.sh overlay --dir DIR --no-mesh            # 只画质心
#   ./scripts/dive.sh extract --frames F --no-bezier-smooth  # 关闭曲线平滑
#   ./scripts/dive.sh extract --frames F --no-gb             # 不采国标那条曲线
#   ./scripts/dive_batch.sh DIR --skip 123529                # 整目录批量
#   ./scripts/dive_summary.py --csv outputs/dive_summary.csv  # 跨 clip 汇总
#
# extract 会把源路径写进 curve.json，之后各步只给 --dir 就够了；不给 --dir 时
# 取 outputs/ 下最新的那个 dive_* 目录。
#
# 尺度（--scale）三选一，见 python/dive/align.py 顶部的说明：
#   metrics  分析器标定 0.008157 m/px，腾空 g=-9.13（默认，独立于 HSMR）
#   hsmr     HSMR 自己的深度 0.010804，体高自洽但 g=-12.1
#   gravity  反解出 g=-9.81 的那个值，用它报 g 是循环论证，只作对照
#
# 曲线平滑（默认开）：抛物线拟合之后对 mesh/gb 做轻度贝塞尔重采样，再进
# plot/render/overlay。--no-bezier-smooth 关闭；--bezier-ctrl N 调控制点数。
#
# 不需要 FBX SDK。完整选项：python -m python.dive --help
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/../swim_fbx_demo/.venv/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python3)"

if (($# < 1)) || [[ "${1:-}" == --help || "${1:-}" == -h || "${1:-}" == help ]]; then
  sed -n '2,31p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
  [[ "${1:-}" == --help || "${1:-}" == -h || "${1:-}" == help ]] && exit 0
  exit 2
fi

cd "$ROOT"
exec "$PY" -m python.dive "$@"
