"""三角网格的体积、重心、闭合性、体素化。纯 numpy。

体积重心走散度定理：把闭合网格看成均质实体，每个三角形与原点张成一个四面体，
带符号体积加权其质心求和。

    V = Σ (a · (b × c)) / 6
    C = Σ (a+b+c)/4 · (a · (b × c)) / Σ (a · (b × c))

要求网格闭合且法向一致。闭合不等于法向一致：翻转任意一个三角形的绕向，
`is_closed` 仍然通过而体积会错掉那个四面体的两倍——所以 `tramp.verify` 里那条
"法向翻转会被体积察觉"的断言不是多余的，它守的正是 `is_closed` 管不到的那一半。
"""
import numpy as np


def volume_centroid(vertices, faces):
    """(体积, 重心)。体积单位是坐标单位的三次方。"""
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    volume6 = np.einsum("ij,ij->i", a, np.cross(b, c))
    total = volume6.sum()
    if abs(total) < 1e-9:
        raise ValueError("退化网格：带符号体积约为 0")
    return total / 6.0, ((a + b + c) / 4.0 * volume6[:, None]).sum(0) / total


def edge_owners(faces):
    """每条无向边 -> 共享它的三角形下标列表。"""
    owners = {}
    for index, triangle in enumerate(faces):
        for k in range(3):
            i, j = int(triangle[k]), int(triangle[(k + 1) % 3])
            owners.setdefault((i, j) if i < j else (j, i), []).append(index)
    return owners


def is_closed(faces):
    """(是否闭合, 边数, 单边最多被几个面共享)。闭合流形要求恒为 2。"""
    counts = np.array([len(v) for v in edge_owners(faces).values()])
    return bool((counts == 2).all()), len(counts), int(counts.max())


def weighted_centre(positions, masses):
    """质量加权重心。positions (N, 3)，masses (N,)。"""
    total = masses.sum()
    if total <= 0:
        raise ValueError("总质量必须为正")
    return (positions * masses[:, None]).sum(0) / total


def interior_voxels(vertices, faces, pitch, axis=0):
    """网格内部的规则网格点，(N, 3)。沿 `axis` 打射线，奇偶法判内外。

    交点排序、去重后成对取"进-出"区间，落在区间内的格点即内部。去重是必需的：
    射线擦过两个三角形的公共边时两边各报一个交点，不合并就凑成奇数被整行丢掉——
    规则形体（立方体的对角线方向）能因此损失 25% 的体素。合并后仍是奇数说明射线
    真的相切于某个角，跳过该行比猜一个区间安全。

    体素化而不是凸包切分：凸包会把腋下、裆部的空腔算成实体，各段凸包又彼此重叠，
    实测总体积从 63.0 L 掉到 29.6 L（既漏又重）。体素法在 1.2 cm 下总体积对整网格
    只差 +0.26%。
    """
    lateral = [k for k in range(3) if k != axis]
    low, high = vertices.min(0) - pitch, vertices.max(0) + pitch
    along = np.arange(low[axis] + pitch / 2, high[axis], pitch)
    grids = [np.arange(low[k] + pitch / 2, high[k], pitch) for k in lateral]
    rays = np.stack(np.meshgrid(*grids, indexing="ij"), -1).reshape(-1, 2)

    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    u, v = lateral
    # 在 (u, v) 投影面上做重心坐标测试。与射线平行的面 det=0，会算出 nan，
    # 被下面的 isfinite 排除——那正是"这个面不挡这条射线"。
    det = ((b[:, u] - a[:, u]) * (c[:, v] - a[:, v])
           - (c[:, u] - a[:, u]) * (b[:, v] - a[:, v]))
    tolerance = pitch * 1e-6

    found = []
    with np.errstate(all="ignore"):
        for start in range(0, len(rays), 400):
            batch = rays[start:start + 400]
            ru, rv = batch[:, 0][:, None], batch[:, 1][:, None]
            l1 = ((ru - a[:, u]) * (c[:, v] - a[:, v])
                  - (c[:, u] - a[:, u]) * (rv - a[:, v])) / det
            l2 = ((b[:, u] - a[:, u]) * (rv - a[:, v])
                  - (ru - a[:, u]) * (b[:, v] - a[:, v])) / det
            hit = ((l1 >= 0) & (l2 >= 0) & (l1 + l2 <= 1)
                   & np.isfinite(l1) & np.isfinite(l2))
            crossing = (a[:, axis] + l1 * (b[:, axis] - a[:, axis])
                        + l2 * (c[:, axis] - a[:, axis]))
            for row in range(len(batch)):
                hits = np.sort(crossing[row][hit[row]])
                if len(hits) > 1:
                    hits = hits[np.concatenate(
                        [[True], np.diff(hits) > tolerance])]
                if len(hits) < 2 or len(hits) % 2:
                    continue
                for pair in range(0, len(hits), 2):
                    inside = along[(along > hits[pair]) & (along < hits[pair + 1])]
                    if not len(inside):
                        continue
                    block = np.empty((len(inside), 3))
                    block[:, axis] = inside
                    block[:, u] = batch[row, 0]
                    block[:, v] = batch[row, 1]
                    found.append(block)
    if not found:
        raise ValueError("体素化没找到任何内部点")
    return np.vstack(found)
