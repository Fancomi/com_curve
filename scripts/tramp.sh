#!/usr/bin/env bash
# 蹦床 FBX 线的统一入口 —— 从 BengChuang_01.fbx 提取重心曲线、反算根骨骼。
#
#   ./scripts/tramp.sh STEPS [选项…]
#
# STEPS 逗号分隔、按给出顺序执行：
#   extract   FBX -> outputs/tramp/curves.{csv,json,npz} + markers.fbx
#             五种重心口径一次算完：root(骨盆) / mesh(体积重心) /
#             mujoco(14刚体·网格质量) / anthro(14刚体·Dempster质量) /
#             gb(14刚体·GB/T 17245 国标质量)
#   retarget  反算根骨骼，让被盯的曲线走完美抛物线
#             -> outputs/tramp/retarget/<target>.{fbx,csv,json,npz}
#             --targets root 是 before 基线（原设计的意图，但用精确 g）
#   plot      CSV -> curves.png / trajectory.png，以及每个 retarget 产物一张图
#   render    四格对照视频 -> outputs/tramp/render_<A>_<B>.mp4
#   verify    独立校验：解析体、蒙特卡洛、体段恒等式、反算产物
#
# 常用：
#   ./scripts/tramp.sh extract,plot                      # 五条曲线 + 图
#   ./scripts/tramp.sh render                            # root vs mesh 视频
#   ./scripts/tramp.sh render --pair mujoco anthro       # 另两种口径
#   ./scripts/tramp.sh retarget --targets root mesh mujoco anthro gb
#   ./scripts/tramp.sh verify
#
# 源 FBX 默认找 inputs/BengChuang_01.fbx，其次仓库上一级；用 --src 显式指定。
# 全部逻辑在 python/tramp/，这个脚本只挑解释器并原样转发参数。
# 完整选项：python -m python.tramp --help
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# FBX Python SDK 只有 cp310 的轮子，所以固定用 swim_fbx_demo 那个环境。
PY="$ROOT/../swim_fbx_demo/.venv/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python3)"

if (($# < 1)) || [[ "${1:-}" == --help || "${1:-}" == -h || "${1:-}" == help ]]; then
  # 用法说明只有一份：把本文件顶部的注释块打出来，不再复制一遍。
  sed -n '2,26p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
  [[ "${1:-}" == --help || "${1:-}" == -h || "${1:-}" == help ]] && exit 0
  exit 2
fi

cd "$ROOT"
exec "$PY" -m python.tramp "$@"
