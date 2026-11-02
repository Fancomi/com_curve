"""整体旋转修复：让每段腾空的全身角动量守恒。

`retarget` 只管质心走抛物线；这里管的是**整体朝向**。原动画在主视角（相机 +Z，看空翻
轴）下已经很好，换到正视/俯视就显出生硬：转体是直接拧根骨骼拧出来的——双臂抬举角
左右逐帧完全相等，空中没有任何不对称动作能产生转体，却在 2 帧内凭空起转。物理上
腾空只受重力，重力过质心、不产生力矩，所以绕质心的角动量 L 必须恒定；原动画的 L
在转体段偏离空翻轴最多 55°，离面分量在 ±50 kg·m²/s 之间乱摆。

修法：每段腾空求一条逐帧整体旋转 θ(t)，人体绕被盯曲线（默认 mesh 质心）整体转，
使修正后的 L 在整段腾空里是一个常量（三个分量都放开，常量本身也是未知数）。

  - 绕质心转：质心不动，`retarget` 做好的抛物线原样保留。
  - 两端 θ = 0：触床帧姿态与脚的位置逐位不变。
  - 关节动作一点不改，只改 `Root_M` 的 R/T 曲线——父链恒等、全部蒙皮骨都挂在它下面，
    所以"整体转 Q" ≡ "根骨骼左乘 Q，根位置绕枢轴转"。

求解是高斯-牛顿 + 有限差分雅可比，残差 = [L(θ) − L₀ ; √λ·θ 的二阶差分]。修正量大
（转体段上百度），一步到位会跳进别的分支，所以 λ 从 1e-1 逐级放到 1e-4（continuation）。
scipy 的 trf/lm 在这里第一步就因 xtol 停下、什么都不改，手写的三步收敛。

试过而放弃的：只修离面两个轴（保留原空翻节奏，但空翻分量本身在腾空中涨 2.5 倍，
仍不守恒）、追平滑后的原曲线（只削尖峰，离面偏差剩一半）。用户选了全守恒。

角动量用 14 段体素刚体（`bodies.RigidBodies`，ρ = 1000）算：段质心 + 绑定惯量逐帧刚体
变换。躯干整段挂在 Root_M 下、脊柱弯曲不计，所以团身时空翻分量的波动会被高估一些。

除 `sample` 外纯 numpy；单位**米 / 千克 / 秒**（L 是 kg·m²/s），场景的厘米在
`sample` / `body_of` 里换算。
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot

# 平滑权重的 continuation 序列，最后一级就是最终的 λ。
LAMBDAS = (1e-1, 1e-2, 1e-3, 1e-4)


def body_of(bodies):
    """`bodies.RigidBodies`（cm）-> `Body`（m）。"""
    return Body(bodies.mass_kg, bodies.bind_centre / 100.0, bodies.bind_inertia * 1e-4)


def sample(bodies, stamps):
    """逐帧各段的 (旋转 (n,S,3,3), 平移 (n,S,3) m)。"""
    pairs = [bodies.transforms(stamp) for stamp in stamps]
    return (np.array([r for r, _t in pairs]),
            np.array([t for _r, t in pairs]) / 100.0)


def lateral_axis(bodies):
    """绑定姿态的骨盆侧向轴：左右大腿段质心连线（单位向量）。"""
    names = list(bodies.names)
    axis = (bodies.bind_centre[names.index("thigh_L")]
            - bodies.bind_centre[names.index("thigh_R")])
    return axis / np.linalg.norm(axis)


class Body:
    """14 段的质量 (kg)、绑定质心 (m)、绑定惯量（绕段质心，kg·m²）。"""

    def __init__(self, mass, centre, inertia):
        self.mass = np.asarray(mass, dtype=np.float64)
        self.centre = np.asarray(centre, dtype=np.float64)
        self.inertia = np.asarray(inertia, dtype=np.float64)
        self.total = float(self.mass.sum())

    def centres(self, R, t):
        """各段世界质心 (n, S, 3)。R 是列向量约定的旋转，t 是平移 (m)。"""
        return np.einsum("nsij,sj->nsi", R, self.centre) + t

    def momentum(self, R, C, dt):
        """(绕全身质心的 L (n, 3), 全身质心 (n, 3))。

        L = Σ m (r × v) + Σ I_world ω：前一项是段质心绕全身质心的公转，后一项是各段
        自转。速度一律中心差分——用解析的子帧导数会把 cubic Euler 键间的甩圈算进来，
        而 25 fps 播放时根本求不到那些位置。
        """
        m = self.mass
        com = (C * m[None, :, None]).sum(1) / self.total
        V = np.gradient(C, dt, axis=0)
        vc = np.gradient(com, dt, axis=0)
        world_inertia = np.einsum("nsij,sjk,nslk->nsil", R, self.inertia, R)
        r = C - com[:, None]
        L = ((m[None, :, None] * np.cross(r, V - vc[:, None])).sum(1)
             + np.einsum("nsij,nsj->ni", world_inertia, segment_omega(R, dt)))
        return L, com


def segment_omega(R, dt):
    """各段世界角速度 (n, S, 3)，中心差分；两端单侧。"""
    n = len(R)
    after = np.concatenate([R[1:2], R[2:], R[-1:]])
    before = np.concatenate([R[:1], R[:-2], R[-2:-1]])
    span = np.full(n, 2 * dt)
    span[0] = span[-1] = dt
    relative = np.einsum("nsij,nskj->nsik", after, before)
    flat = Rot.from_matrix(relative.reshape(-1, 3, 3)).as_rotvec()
    return flat.reshape(n, R.shape[1], 3) / span[:, None, None]


def rotate(R, C, pivot, theta):
    """整体绕逐帧枢轴转 theta（世界系旋转向量）。返回 (R', C')。"""
    Q = Rot.from_rotvec(theta).as_matrix()
    return (np.einsum("nij,nsjk->nsik", Q, R),
            np.einsum("nij,nsj->nsi", Q, C - pivot[:, None]) + pivot[:, None])


def plane_basis(normal):
    """与 normal 垂直的两个单位向量 (2, 3)。"""
    normal = normal / np.linalg.norm(normal)
    helper = np.array([1.0, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1.0, 0])
    first = np.cross(normal, helper)
    first /= np.linalg.norm(first)
    return np.stack([first, np.cross(normal, first)])


def gauss_newton(residual, x, iters=6, step=1e-6):
    for _ in range(iters):
        r = residual(x)
        J = np.empty((len(r), len(x)))
        for j in range(len(x)):
            probe = x.copy()
            probe[j] += step
            J[:, j] = (residual(probe) - r) / step
        dx = np.linalg.lstsq(J, r, rcond=None)[0]
        x = x - dx
        if np.abs(dx).max() < 1e-9:
            break
    return x


def solve_flight(body, R, C, pivot, dt, lambdas=LAMBDAS):
    """一段腾空的 θ (n, 3)，两端为 0。R/C 是这段的段旋转与段质心，pivot 是枢轴轨迹。"""
    n = len(R)
    L0, _com = body.momentum(R, C, dt)

    def unpack(x):
        theta = np.zeros((n, 3))
        theta[1:-1] = x[:3 * (n - 2)].reshape(-1, 3)
        return theta

    x = np.zeros(3 * (n - 2) + 3)
    x[-3:] = L0.mean(0)
    for lam in lambdas:
        def residual(x, lam=lam):
            theta = unpack(x)
            L, _c = body.momentum(*rotate(R, C, pivot, theta), dt)
            acc = (theta[2:] - 2 * theta[1:-1] + theta[:-2]).ravel() / dt ** 2
            return np.concatenate([(L - x[-3:][None]).ravel(), np.sqrt(lam) * acc])
        x = gauss_newton(residual, x)
    return unpack(x)


def solve(body, R, C, pivot, frames, windows, dt):
    """全片 θ (N, 3)：各腾空段独立求，触床帧为 0。返回 (θ, 逐段报告)。"""
    theta = np.zeros((len(frames), 3))
    report = []
    L_before, _c = body.momentum(R, C, dt)
    for start, stop in windows:
        a, b = int(np.searchsorted(frames, start)), int(np.searchsorted(frames, stop))
        part = slice(a, b + 1)
        theta[part] = solve_flight(body, R[part], C[part], pivot[part], dt)
        L_after, _c = body.momentum(*rotate(R[part], C[part], pivot[part],
                                            theta[part]), dt)
        report.append({
            "frames": [int(start), int(stop)],
            "correction_max_deg": round(float(np.degrees(
                np.linalg.norm(theta[part], axis=1)).max()), 2),
            "L_before_spread": round(spread(L_before[part]), 3),
            "L_after_spread": round(spread(L_after), 3),
            "L_after": [round(float(v), 3) for v in L_after[1:-1].mean(0)],
        })
    return theta, report


def spread(L):
    """L 偏离段均值的平均幅度（kg·m²/s），两端各丢一帧（单侧差分不可信）。守恒 = 0。"""
    inner = L[1:-1]
    return float(np.linalg.norm(inner - inner.mean(0), axis=1).mean())


def unwrap_euler(angles, previous):
    """把一组 extrinsic xyz Euler 角换成与 previous 最近的等价表示（度）。

    同一个旋转有两族解 (x, y, z) 与 (x+180, 180−y, z+180)，每个分量再差 360 的整数倍。
    原文件 Root_M 有 33 处相邻键差 >90°（最大 180°），cubic 插值会在这些键之间甩出整圈
    ——整数帧上看不见，引擎重采样或运动模糊时就露出来，所以写回时顺手展开。
    """
    x, y, z = angles
    best, score = None, np.inf
    for candidate in (np.array([x, y, z]), np.array([x + 180.0, 180.0 - y, z + 180.0])):
        shifted = candidate + 360.0 * np.round((previous - candidate) / 360.0)
        distance = np.abs(shifted - previous).sum()
        if distance < score:
            best, score = shifted, distance
    return best


def root_keys(theta, euler, translation, pre_rotation, pivot_cm):
    """新的 Root_M 键：(Euler 度 (N,3), 平移 cm (N,3))。

    Root_M 世界旋转 = PreRotation · Euler(key)（父链恒等），整体左乘 Q 之后新键就是
    Pre⁻¹ · Q · Pre · Euler(key)；平移绕枢轴转：t' = Q (t − p) + p。
    """
    pre = Rot.from_euler("xyz", pre_rotation, degrees=True)
    Q = Rot.from_rotvec(theta)
    fixed = (pre.inv() * Q * pre * Rot.from_euler("xyz", euler, degrees=True)
             ).as_euler("xyz", degrees=True)
    out = np.empty_like(fixed)
    out[0] = unwrap_euler(fixed[0], euler[0])
    for k in range(1, len(fixed)):
        out[k] = unwrap_euler(fixed[k], out[k - 1])
    moved = np.einsum("nij,nj->ni", Q.as_matrix(), translation - pivot_cm) + pivot_cm
    return out, moved


def traces(body, R, C, lateral, dt):
    """画图/视频用的逐帧量：L (n,3)、侧向轴与世界 Z 的夹角（°）、躯干角速度（°/s）。

    侧向轴 = 左右大腿段的绑定质心连线跟着躯干转：±90° 是纯空翻，扫过 0 就是在转体。
    躯干段挂在 Root_M 上，所以它的角速度就是根骨骼的世界角速度。
    """
    L, _c = body.momentum(R, C, dt)
    axis = R[:, 0] @ lateral
    tilt = np.degrees(np.arcsin(np.clip(axis[:, 2], -1.0, 1.0)))
    rate = np.degrees(np.linalg.norm(segment_omega(R[:, :1], dt)[:, 0], axis=1))
    return L, tilt, rate
