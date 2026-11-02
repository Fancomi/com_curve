"""在"某条曲线走完美抛体轨迹"的前提下反算根骨骼应有的位置。

原设计把**根骨骼**（`Root_M`，骨盆）按抛物线去 K，但抛体运动约束的是**质心**而不是
骨盆——空中挥臂屈体时两者相差可达 26 cm。所以原动画的质心并不走抛物线：实测腾空段
拟合出 −8.77 m/s²。

修正靠一个恰好成立的性质：**平移根骨骼会把整个人刚体平移，被盯的那条曲线随之平移
同样的量。** `Root_M` 的父链（`WLN/DeformationSystem`）全片是恒等变换，pivot 与偏移
全为零，所以 `LclTranslation` 就是世界位置。实测验证过：给三条 T 曲线各加常量 Δ，
逐帧质心位移与 Δ 逐位相等（5875 个顶点最大偏差 2e-7 cm，纯浮点噪声）。于是修正量是
闭式的，不需要迭代：

    Δroot(t) = 目标轨迹(t) − 被盯曲线的实际位置(t)

`target` 选盯哪条曲线：`mesh`（体积重心）或 `gb`（国标）。两者各是一个产物，
顶点高度由腾空时间 T 唯一决定（两端等高时升高 = gT²/8），与盯哪条几乎无关，彼此只差
几厘米——`verify` 的共模检查守的就是这件事。
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
    """把 Δ 加到节点三条 LclTranslation 曲线的关键帧上。"""
    edit_keys(node, "LclTranslation", layer, frames, mode,
              lambda index, column, old: old + float(delta[index, column]))


def read_keys(node, prop, layer, frames, mode):
    """节点一个 TRS 属性三条曲线的键值 (N, 3)。键与帧的对应校验同 `edit_keys`。"""
    out = np.empty((len(frames), 3))

    def grab(index, column, old):
        out[index, column] = old
        return old

    edit_keys(node, prop, layer, frames, mode, grab, write=False)
    return out


def edit_keys(node, prop, layer, frames, mode, update, write=True):
    """逐键改写 `prop`（"LclTranslation" / "LclRotation"）的三条曲线：新值 = update(帧序, 轴, 旧值)。

    关键帧与采样帧一一对应（实测 Root_M 的 T/R 六条曲线各 485 个键、每帧一个、全
    eTangentAuto），所以可以按索引直接改。不成立时宁可报错也不要按时间去猜最近的键——
    错配一帧，修正量会整体偏移一帧，而画面上几乎看不出来。
    """
    for column, axis in enumerate("XYZ"):
        curve = getattr(node, prop).GetCurve(layer, axis, False)
        if curve is None:
            raise ValueError(f"{node.GetName()} 没有 {prop}.{axis} 曲线")
        if curve.KeyGetCount() != len(frames):
            raise ValueError(
                f"{node.GetName()}.{prop}.{axis} 有 {curve.KeyGetCount()} 个键，"
                f"采样了 {len(frames)} 帧，对不上")
        if write:
            curve.KeyModifyBegin()
        try:
            for index in range(curve.KeyGetCount()):
                at = int(curve.KeyGetTime(index).GetFrameCount(mode))
                if at != int(frames[index]):
                    raise ValueError(f"{node.GetName()}.{prop}.{axis} 第 {index} 个键在 "
                                     f"f{at}，采样的是 f{frames[index]}")
                value = update(index, column, curve.KeyGetValue(index))
                if write:
                    curve.KeySetValue(index, float(value))
        finally:
            if write:
                curve.KeyModifyEnd()
