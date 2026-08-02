#!/usr/bin/env bash
# 蹦床线跑一整轮：四条曲线 + 一个 before 基线 + 三个 after + 图 + 三个对比视频。
# 双击或 `./scripts/tramp_full.sh` 即可，无参数。约 4 分钟。
#
# 为什么 before 也要跑一遍反算（--targets 里带 root）：
#   两端钉住 + 精确 g 之后，抛物线顶点由腾空时间 T 唯一决定（两端等高时升高 gT²/8），
#   与盯哪条曲线几乎无关——四条曲线各自钉完，顶点都升到 5.3 m 均值，彼此只差 4~9 cm。
#   拿未修正的原动画当 before，量出的 +75 cm 里绝大部分是 T 这个共模项，不是"改盯
#   质心"的效果。让 before 与三个 after 共用同一组 T 和同一个 g，共模抵消，剩下的
#   十几厘米才是真信号。
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERE=./scripts/tramp.sh

$HERE extract
$HERE retarget --targets root mesh mujoco anthro
$HERE plot
$HERE render                              # root vs mesh
$HERE render --pair mujoco anthro         # 另两种口径，带 14 个体段质心圆

# before/after 对比：--compare 必须跟着 target 走，否则两格里都是"没被约束成抛物线
# 的那条曲线"，看不出这一版改了什么。
for target in mesh mujoco anthro; do
  $HERE render --src "outputs/tramp/retarget/$target.fbx" \
    --reference outputs/tramp/retarget/root.fbx --compare "$target" \
    --labels "before: 骨盆=抛物线" "after: $target 质心=抛物线"
done

$HERE verify
