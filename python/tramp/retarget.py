"""在"某条曲线走完美抛体轨迹"的前提下反算根骨骼应有的位置。

原设计把**根骨骼**（`Root_M`，骨盆）按抛物线去 K，但抛体运动约束的是**质心**而不是
骨盆——空中挥臂屈体时两者相差可达 26 cm。所以原动画的质心并不走抛物线：实测腾空段
拟合出 −8.77 m/s²，而骨盆是 −9.10。

修正靠一个恰好成立的性质：**平移根骨骼会把整个人刚体平移，被盯的那条曲线随之平移
同样的量。** `Root_M` 的父链（`WLN/DeformationSystem`）全片是恒等变换，pivot 与偏移
全为零，所以 `LclTranslation` 就是世界位置。实测验证过：给三条 T 曲线各加常量 Δ，
逐帧质心位移与 Δ 逐位相等（5875 个顶点最大偏差 2e-7 cm，纯浮点噪声）。于是修正量是
闭式的，不需要迭代：

    Δroot(t) = 目标轨迹(t) − 被盯曲线的实际位置(t)

`target` 选盯哪条曲线，这决定产物的含义：

    root                原设计师的意图，但用精确 g。**它是 before 基线。**
    mesh/mujoco/anthro/gb   四种真质心口径，各是一个 after。

**为什么 before 必须也做一遍。** 两端钉住 + 精确 g 之后，抛物线高度由腾空时间 T 唯一
决定（两端等高时顶点升高 = gT²/8），与盯哪条曲线几乎无关：实测五条曲线各自钉完，
顶点都升到 5.3 m 均值，彼此只差 4~9 cm。所以拿"未修正的原动画"当 before 去比 after，
看到的 +75 cm 里绝大部分是 T 这个共模项。用 `--target root` 产出的 before 与四个
after 共用同一组 T 和同一个 g，共模抵消，剩下的差（|Δroot| 均值 10~11 cm）才是真正
由"盯骨盆 vs 盯质心"造成的。
"""
import numpy as np

from ..common import signal as Sig
from . import curves as Cv
from . import scene as F

GRAVITY_CM = 981.0          # cm/s²，标准重力


def solve(track, frames, contacts, fps, gravity=GRAVITY_CM, min_length=5):
    """(Δroot, 每段窗口的诊断)。Δ 在触床帧上恒为 0。

    腾空窗口带上首末两段（`include_edges`）：这个文件开头 11 帧就是从空中开始的，
    漏掉它等于那段完全没被修正。太短的段（<5 帧）不改——三点以下拟不出抛物线。
    """
    dt = 1.0 / fps
    delta = np.zeros_like(track)
    report = []
    windows = Sig.flights_from_contacts(contacts, True, frames)
    for start, stop in windows:
        a = int(np.searchsorted(frames, start))
        b = int(np.searchsorted(frames, stop))
        if b - a + 1 < min_length:
            continue
        actual = track[a:b + 1]
        target = Sig.parabola_through_ends(actual, dt, gravity)
        delta[a:b + 1] = target - actual
        shift = np.linalg.norm(target - actual, axis=1)
        report.append({
            "frames": [int(frames[a]), int(frames[b])],
            "samples": int(b - a + 1),
            "shift_mean_cm": round(float(shift.mean()), 3),
            "shift_max_cm": round(float(shift.max()), 3),
            "apex_before_m": round(float(actual[:, 1].max() / Cv.CM_PER_M), 3),
            "apex_after_m": round(float(target[:, 1].max() / Cv.CM_PER_M), 3),
            "takeoff_before_mps": round(
                float((actual[1, 1] - actual[0, 1]) / dt / Cv.CM_PER_M), 3),
            "takeoff_after_mps": round(
                float((target[1, 1] - target[0, 1]) / dt / Cv.CM_PER_M), 3),
        })
    return delta, report


def check_root_chain(root_bone, mode, frames):
    """根链必须是"平移 root = 平移全身"，否则闭式解不成立。

    父链有旋转或缩放、pivot 非零，都会让这个前提破掉——宁可报错也不要算出个看着
    像的结果。抽三帧够了：这三样都是静态属性或恒等变换。
    """
    for frame in (frames[0], frames[len(frames) // 2], frames[-1]):
        stamp = F.time_at(frame, mode)
        local = root_bone.EvaluateLocalTranslation(stamp)
        world = F.node_translation(root_bone, stamp)
        gap = max(abs(world[k] - local[k]) for k in range(3))
        if gap > 1e-6:
            raise SystemExit(
                f"f{frame}: {root_bone.GetName()} 的局部平移与世界位置差 {gap:.3e} cm，"
                f"父链不是恒等变换，闭式解不适用")


def apply(node, layer, frames, delta, mode):
    """把 Δ 加到节点三条 LclTranslation 曲线的关键帧上。

    关键帧与采样帧一一对应（实测这三条曲线各 485 个键、每帧一个、全 eTangentAuto），
    所以可以按索引直接加。不成立时宁可报错也不要按时间去猜最近的键——错配一帧，
    修正量会整体偏移一帧，而画面上几乎看不出来。
    """
    for column, axis in enumerate("XYZ"):
        curve = node.LclTranslation.GetCurve(layer, axis, False)
        if curve is None:
            raise ValueError(f"{node.GetName()} 没有 LclTranslation.{axis} 曲线")
        if curve.KeyGetCount() != len(frames):
            raise ValueError(
                f"{node.GetName()}.{axis} 有 {curve.KeyGetCount()} 个键，"
                f"采样了 {len(frames)} 帧，对不上")
        curve.KeyModifyBegin()
        for index in range(curve.KeyGetCount()):
            at = int(curve.KeyGetTime(index).GetFrameCount(mode))
            if at != int(frames[index]):
                curve.KeyModifyEnd()
                raise ValueError(f"{node.GetName()}.{axis} 第 {index} 个键在 f{at}，"
                                 f"采样的是 f{frames[index]}")
            curve.KeySetValue(index,
                              curve.KeyGetValue(index) + float(delta[index, column]))
        curve.KeyModifyEnd()
