#!/usr/bin/env bash
# 蹦床线跑一整轮：两条重心曲线 + 两个反算产物 + 旋转修复 + 图 + 对比视频。
# 双击或 `./scripts/tramp_full.sh` 即可，无参数。约 6 分钟。
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERE=./scripts/tramp.sh

$HERE extract
$HERE retarget --targets mesh gb
$HERE spin                                # 输入 retarget/mesh.fbx
$HERE plot
$HERE render                              # mesh vs gb，带 14 个体段质心圆

# 反算前后：--compare 必须跟着 target 走，否则两格里都是"没被约束成抛物线的那条
# 曲线"，看不出这一版改了什么。
for target in mesh gb; do
  $HERE render --src "outputs/tramp/retarget/$target.fbx" \
    --reference inputs/BengChuang_01.fbx --compare "$target" \
    --labels "before: 原动画" "after: $target 质心=抛物线"
done
$HERE spinvideo                           # 旋转修复前后，跟踪质心三视角

$HERE verify
