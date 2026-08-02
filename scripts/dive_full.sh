#!/usr/bin/env bash
# 跳水线跑一整轮：提取 + 三张图 + 正交四格视频 + 叠加视频。
#
#   ./scripts/dive_full.sh FRAMES.bin.br
#   ./scripts/dive_full.sh                # 不给参数就用 outputs 下最新那个 clip
#
# 约 3 分钟，其中 overlay 占大头（599 帧 1920×1080 底图）。
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERE=./scripts/dive.sh

if (($# >= 1)); then
  $HERE extract --frames "$1" --check
  # extract 把源路径写进了 curve.json，后面各步认最新那个目录就够了。
  $HERE plot,render,overlay
else
  $HERE plot,render,overlay
fi
