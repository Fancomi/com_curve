#!/usr/bin/env bash
# 跳水线跑一整轮：提取 + 三张图 + 正交六格视频 + 叠加视频。
#
#   ./scripts/dive_full.sh FRAMES.bin.br
#   ./scripts/dive_full.sh                # 不给参数就用 outputs 下最新那个 clip
#
# 约 3 分钟，其中 overlay 占大头（599 帧 1920×1080 底图）。
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERE=./scripts/dive.sh

if (($# >= 1)); then
  frames="$1"
  $HERE extract --frames "$frames" --check
  # 用 frames 文件名锁定输出目录，避免 outputs/ 里已有别的 dive_* 时 latest 指错。
  stem="$(basename "$frames")"
  stem="${stem%%_replay_frames*}"
  $HERE plot,render,overlay --dir "outputs/dive_${stem}"
else
  $HERE plot,render,overlay
fi
