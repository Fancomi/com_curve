"""曲线上的信号处理：导数、抛物线拟合、连续段、理想抛体轨迹。纯 numpy。

单位无关：传 cm 得 cm/s²，传 m 得 m/s²。调用方负责把单位对齐，别混。
"""
import numpy as np


def runs_of(mask):
    """布尔序列里连续 True 的 (起, 止) 闭区间下标。"""
    out, index = [], 0
    while index < len(mask):
        if mask[index]:
            end = index
            while end + 1 < len(mask) and mask[end + 1]:
                end += 1
            out.append((index, end))
            index = end + 1
        else:
            index += 1
    return out


def derivative(series, dt):
    """中心差分一阶导，两端单侧。列方向独立处理。

    首末帧只有一阶精度，画图时不要拿它们下结论。
    """
    return np.gradient(series, dt, axis=0, edge_order=2)


def masked_derivative(series, valid, dt):
    """只在连续有效段内求导，段的首末帧置 nan。

    直接把无效帧填 0 再求导会在有效/无效边界造出 ±100 m/s 的假尖峰——dive 那条线
    的 CSV 里就出现过 f437 的 +87.8 m/s（那正是最后一个有效帧）。边界帧没有可用的
    邻居，nan 是唯一诚实的答案。
    """
    out = np.full(np.shape(series), np.nan, dtype=np.float64)
    for start, stop in runs_of(np.asarray(valid, dtype=bool)):
        if stop - start < 2:
            continue
        out[start + 1:stop] = derivative(series[start:stop + 1], dt)[1:-1]
    return out


def fit_parabola(height, dt):
    """(加速度, 残差 rms)。对一段高度序列拟合二次曲线。

    腾空段的加速度应当接近 -g。这是对整条链路的独立校验：骨架求值、蒙皮、
    三角化、尺度标定任何一处错了，这个数就不对。
    """
    height = np.asarray(height, dtype=np.float64)
    if len(height) < 3:
        raise ValueError(f"拟合抛物线至少要 3 个采样点，给了 {len(height)}")
    time = np.arange(len(height)) * dt
    coefficients = np.polyfit(time, height, 2)
    residual = height - np.polyval(coefficients, time)
    return 2.0 * coefficients[0], float(np.sqrt((residual ** 2).mean()))


def flights_from_contacts(contacts, include_edges=False, frames=None):
    """触床区间 -> 腾空区间。默认只取相邻两次触床之间的那些段。

    `include_edges` 额外给出首段（片头到第一次触床）与末段。默认关掉是因为
    「顶点升高」那类统计要求两端都钉在触床上；但产物校验必须打开，否则开局那段
    从空中开始的腾空根本没被检查过。`frames` 给出可用帧范围时才能定出边界段。
    """
    out = [(end + 1, nxt - 1)
           for (_s, end), (nxt, _e) in zip(contacts[:-1], contacts[1:])]
    if include_edges and frames is not None and contacts:
        if contacts[0][0] > frames[0]:
            out.insert(0, (int(frames[0]), contacts[0][0] - 1))
        if contacts[-1][1] < frames[-1]:
            out.append((contacts[-1][1] + 1, int(frames[-1])))
    return [(a, b) for a, b in out if b >= a]


def gravity_over_flights(track, frames, flights, dt, trim=0, unit=1.0):
    """各腾空段的重力拟合汇总。`track` 是 (N, 3) 或 (N,)，`unit` 是"每米几个单位"。

    `trim` 切掉每段两端各若干帧：触床判据是床面被压，而人在离床后的头几帧仍在
    收腿、落地前几帧已在伸腿，那几帧的重心受肢体动作支配而非纯抛体。

    保留 6 位小数而不是 4 位：反算产物的断言要求 |g + 9.81| < 1e-4，round 到 4 位会
    把误差直接压进容差里，那条断言就永远通过了。
    """
    height = np.asarray(track)
    height = height[:, 1] if height.ndim == 2 else height
    per_flight, residuals = [], []
    for start, stop in flights:
        a, b = int(start - frames[0]), int(stop - frames[0]) + 1
        window = height[a:b] / unit
        core = window[trim:len(window) - trim] if trim else window
        gravity, rms = fit_parabola(core, dt)
        per_flight.append(float(gravity))
        residuals.append(rms * unit)
    return {
        "per_flight": [round(v, 6) for v in per_flight],
        "gravity_mean": round(float(np.mean(per_flight)), 6),
        "gravity_std": round(float(np.std(per_flight)), 6),
        "rms_mean": round(float(np.mean(residuals)), 6),
        "rms_max": round(float(np.max(residuals)), 6),
    }


def parabola_through_ends(track, dt, gravity, up=1):
    """两端钉在原位的理想抛体轨迹，形状同 `track`。

    Y（下标 `up`）匀加速 -gravity，其余轴匀速。速度由"起止位置不变"解出：

        y(T) = y0 + v·T - ½gT²   =>   v = (y(T) - y0 + ½gT²) / T

    钉两端而不是整段最小二乘：最小二乘的 |Δ| 小一半，但触床帧上会残留几十厘米，
    脚会悬在空中。钉两端让触床帧 Δ 恒为 0，代价是离床速度跳变——那不是瑕疵，
    要让曲线走真抛物线并落回原处，起跳必须蹬出更大的速度。
    """
    flat = np.asarray(track, dtype=np.float64)
    single = flat.ndim == 1
    flat = flat.reshape(-1, 1) if single else flat
    count = len(flat)
    if count < 3:
        raise ValueError(f"定抛物线至少要 3 个采样点，给了 {count}")
    time = np.arange(count) * dt
    total = time[-1]
    out = np.empty_like(flat)
    for axis in range(flat.shape[1]):
        accel = -gravity if (single or axis == up) else 0.0
        start = flat[0, axis]
        speed = (flat[-1, axis] - start - 0.5 * accel * total ** 2) / total
        out[:, axis] = start + speed * time + 0.5 * accel * time ** 2
    return out.ravel() if single else out
