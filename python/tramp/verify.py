"""独立校验。每一条都用**与被测代码不同的路径**得出参照值。

被删掉的自评式断言（它们只在重复被测代码或检查输入文件的性质）：

  * 网格闭合、权重归一 —— 查的是输入 FBX 的性质，同一个文件跑一万次结果一样。
    这两个数已经记在 `curves.json` 的 `scene` 块里，需要时看那里。
  * Dempster 质量比闭合 —— `segments.dempster_mass_fractions()` 内部已经 raise。
  * 八面体的体积与重心 —— 教科书用例，写对了就永远不会失败。留下的是它的**反向**
    形式（翻转法向必须被察觉），那条才有内容。
  * 段质心离全身重心 < 1.2 × 身高 —— 阈值宽到只能抓"FK 彻底崩了"，而那种情况
    推导点检查已经抓住了；它却要多扫一遍 485 帧。

留下的每一条都对应一个**真实发生过或能真实发生**的失败。
"""
import numpy as np

from ..common import gb17245 as GB
from ..common import geometry as G
from ..common import paths as P
from ..common import signal as Sig
from . import bodies as B
from . import curves as Cv
from . import scene as F
from . import segments as S

PASSED = []


def check(name, ok, detail):
    PASSED.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL':4s}] {name}：{detail}")


def octahedron(radius=3.0):
    """正八面体：体积 4r³/3，重心在原点，三角面朝外。"""
    r = radius
    vertices = np.array([[r, 0, 0], [-r, 0, 0], [0, r, 0], [0, -r, 0],
                         [0, 0, r], [0, 0, -r]], dtype=np.float64)
    faces = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
                      [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]], dtype=np.int64)
    return vertices, faces, 4.0 * r ** 3 / 3.0


def monte_carlo(vertices, faces, samples=400_000, seed=7):
    """(体积, 重心, 内部点数)：包围盒撒点 + 奇偶射线法。

    与散度定理零共享代码，所以能抓到公式本身写错。公差由采样数决定：40 万点在这个
    体型上约 1% 体积误差、0.2 cm 重心误差，足够抓出量级错误（少算一半四边形、绑定
    矩阵搞反），抓不出小数点后两位——后者本来也不该靠随机方法验。
    """
    low, high = vertices.min(0), vertices.max(0)
    points = np.random.default_rng(seed).uniform(low, high, size=(samples, 3))
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    inside = np.zeros(samples, dtype=bool)
    with np.errstate(all="ignore"):
        for start in range(0, samples, 20_000):
            batch = points[start:start + 20_000]
            crossings = np.zeros(len(batch), dtype=np.int64)
            y, z = batch[:, 1][:, None], batch[:, 2][:, None]
            for first in range(0, len(faces), 3000):
                A, Bb, Cc = (a[first:first + 3000], b[first:first + 3000],
                             c[first:first + 3000])
                det = ((Bb[:, 1] - A[:, 1]) * (Cc[:, 2] - A[:, 2])
                       - (Cc[:, 1] - A[:, 1]) * (Bb[:, 2] - A[:, 2]))
                l1 = ((y - A[:, 1]) * (Cc[:, 2] - A[:, 2])
                      - (Cc[:, 1] - A[:, 1]) * (z - A[:, 2])) / det
                l2 = ((Bb[:, 1] - A[:, 1]) * (z - A[:, 2])
                      - (y - A[:, 1]) * (Bb[:, 2] - A[:, 2])) / det
                hit = ((l1 >= 0) & (l2 >= 0) & (l1 + l2 <= 1)
                       & np.isfinite(l1) & np.isfinite(l2))
                x_hit = (A[:, 0] + l1 * (Bb[:, 0] - A[:, 0])
                         + l2 * (Cc[:, 0] - A[:, 0]))
                crossings += (hit & (x_hit > batch[:, 0][:, None])).sum(1)
            inside[start:start + 20_000] = (crossings % 2) == 1
    volume = float(np.prod(high - low) * inside.mean())
    return volume, points[inside].mean(0), int(inside.sum())


def bind_pose_error(node, skin):
    """把每根骨摆回它自己的 `TransformLinkMatrix`，蒙皮必须还原绑定姿态。

    这是最有价值的一条：它用与 `SkinnedMesh.vertices` 不同的路径重建了一遍蒙皮，
    专门守"绑定逆矩阵的乘法顺序"——`link.Inverse() * (reference * geometric)` 搞反
    不会报错，只会让人偏到别处。
    """
    import fbx

    deformer = node.GetMesh().GetDeformer(0, fbx.FbxDeformer.EDeformerType.eSkin)
    geometric = F.geometric_matrix(node)
    reference = fbx.FbxAMatrix()
    deformer.GetCluster(0).GetTransformMatrix(reference)
    want = F.transform_points(skin.base, F.matrix_np(reference * geometric))

    got = np.zeros_like(skin.base)
    for index in range(deformer.GetClusterCount()):
        cluster = deformer.GetCluster(index)
        if cluster.GetControlPointIndicesCount() == 0:
            continue
        link = fbx.FbxAMatrix()
        cluster.GetTransformLinkMatrix(link)
        own = fbx.FbxAMatrix()
        cluster.GetTransformMatrix(own)
        matrix = F.matrix_np(link * (link.Inverse() * (own * geometric)))
        indices = np.array(cluster.GetControlPointIndices(), dtype=np.int64)
        weights = np.array(cluster.GetControlPointWeights())
        got[indices] += (F.transform_points(skin.base[indices], matrix)
                         * weights[:, None])
    return float(np.abs(got - want).max())


def verify_geometry():
    """反向断言：翻转一个三角形的绕向，体积必须察觉。

    `is_closed` 检查不到法向一致性（每条边仍被两个面共享），所以这条守的正是它管不到
    的那一半。这类"要求某个错误被检测到"的断言很少见也很有价值。
    """
    print("几何内核：")
    vertices, faces, want = octahedron()
    flipped = faces.copy()
    flipped[0] = flipped[0][::-1]
    bad, _centre = G.volume_centroid(vertices, flipped)
    check("法向翻转会被体积察觉", abs(bad - want) > 1.0,
          f"翻一个面后体积 {bad:.3f}（正确 {want:.3f}）")


def verify_mesh(sampler, frames):
    """三角化 + 蒙皮 + 体积重心。"""
    print("\n蒙皮与体积重心：")
    node, skin = sampler.body_node, sampler.skin
    mesh = node.GetMesh()
    expected = sum(mesh.GetPolygonSize(p) - 2
                   for p in range(mesh.GetPolygonCount()))
    check("三角化没丢面", len(skin.triangles) == expected,
          f"{mesh.GetPolygonCount()} 多边形 -> {len(skin.triangles)} 三角形"
          f"（只取前三角会静默丢掉每个四边形的一半，体积错一倍）")
    error = bind_pose_error(node, skin)
    check("绑定姿态代数自检", error < 1e-9, f"最大偏差 {error:.3e} cm")

    for frame in frames:
        if not sampler.first <= frame <= sampler.last:
            continue
        world = skin.vertices(sampler.stamp(frame))
        volume, centre = G.volume_centroid(world, skin.triangles)
        sampled, sampled_centre, count = monte_carlo(world, skin.triangles)
        check(f"f{frame} 蒙特卡洛体积", abs(sampled - volume) / volume < 0.03,
              f"散度 {volume / 1000:.3f} L vs 采样 {sampled / 1000:.3f} L"
              f"（差 {abs(sampled - volume) / volume * 100:.2f}%，内部点 {count}）")
        check(f"f{frame} 蒙特卡洛重心",
              np.linalg.norm(sampled_centre - centre) < 1.0,
              f"差 {np.linalg.norm(sampled_centre - centre):.3f} cm")


def verify_bodies(sampler):
    """体段刚体：体素划分与两种独立划分的合成恒等式。"""
    print("\n体段刚体：")
    bodies = sampler.bodies
    vertices = bodies.bind_vertices
    mesh_volume, mesh_centre = G.volume_centroid(vertices, sampler.skin.triangles)
    error = abs(bodies.voxel_volume_cm3 - mesh_volume) / mesh_volume
    check("体素总体积", error < 0.02,
          f"{bodies.voxel_volume_cm3 / 1000:.3f} L vs 整网 "
          f"{mesh_volume / 1000:.3f} L（差 {error * 100:.2f}%）")

    # 合成恒等式：均质下把体积切开再按体积加权合成，必须还原原始质心。它与划分方式
    # 无关，所以两种完全独立的划分都要过——只测一种的话，过了也可能只是那种划分恰好
    # 自洽。骨线距离版给出的躯干占比差 4.5 个百分点，是真正独立的第二条路径。
    composite = G.weighted_centre(bodies.bind_centre, bodies.mass_kg)
    check("段质心合成回整网重心（蒙皮权重划分）",
          np.linalg.norm(composite - mesh_centre) < 0.2,
          f"差 {np.linalg.norm(composite - mesh_centre):.4f} cm")

    voxels = G.interior_voxels(vertices, sampler.skin.triangles, bodies.pitch)
    which = B.assign_by_bone_lines(voxels, bodies.bind_joints)
    cell = bodies.pitch ** 3
    masses = np.array([(which == i).sum() * cell for i in range(len(bodies.names))])
    centres = np.array([voxels[which == i].mean(0)
                       for i in range(len(bodies.names))])
    other = G.weighted_centre(centres, masses)
    check("段质心合成回整网重心（骨线距离划分）",
          np.linalg.norm(other - mesh_centre) < 0.2,
          f"差 {np.linalg.norm(other - mesh_centre):.4f} cm；两种划分的躯干占比 "
          f"{bodies.volume_cm3[0] / bodies.voxel_volume_cm3 * 100:.1f}% vs "
          f"{masses[0] / masses.sum() * 100:.1f}%")

    # 站姿重心高度占身高：抓的是"体段表接错骨"。文献站姿约 55–57%，这个绑定是抬臂
    # T-pose 会偏高，所以放宽到 62%。接错一整段，比例会立刻掉出区间。
    joints = {**bodies.bind_joints, **S.derive_in_bind(bodies.bind_joints)}
    centres = S.anthropometric_centres(joints)
    anthro = G.weighted_centre(
        np.array([centres[name] for name in bodies.names]), bodies.anthro_mass)
    ratio = (anthro[1] - bodies.floor_cm) / bodies.height_cm
    check("anthro 站姿重心高度占身高", 0.55 <= ratio <= 0.62,
          f"{ratio * 100:.2f}%（文献站姿约 55–57%，抬臂 T-pose 偏高）；同姿态 mesh "
          f"{(mesh_centre[1] - bodies.floor_cm) / bodies.height_cm * 100:.2f}%，"
          f"mujoco {(composite[1] - bodies.floor_cm) / bodies.height_cm * 100:.2f}%")

    # 国标口径的独立核对。三条，各守不同的东西：
    #
    # 1) 站姿重心高度对上**国标自己的表 7「整体质心」行**。那一行给的是"距头顶点占
    #    身高的百分比"（男 43.8 / 女 44.5），换成距地面就是 56.2 / 55.5——这是国标
    #    自带的、与分段实现无关的基准，比抄文献的 55–57% 硬。放宽到 ±6 个百分点，
    #    因为这个绑定是抬臂 T-pose（同姿态下 mesh 58.2%、anthro 59.9% 都偏高）。
    #    注意必须减掉 floor_cm：这个模型脚底在 y≈10.6 而不是 0。
    # 2) 男女表值不同，互换 sex 必须让全身重心真的动。
    # 3) 表 2/3 的绝对 mm 值能反解出全躯干质心比例，与 GB_TRUNK_RATIO 对上——
    #    守的是"合并躯干时随手取了上躯干比例"那个错误（差 10 个百分点，2.8 cm）。
    for sex, label, want in ((S.GB_MALE, "男", S.GB_WHOLE_BODY_FROM_VERTEX[S.GB_MALE]),
                             (S.GB_FEMALE, "女", S.GB_WHOLE_BODY_FROM_VERTEX[S.GB_FEMALE])):
        centre = G.weighted_centre(
            np.array([S.gb_centres(joints, sex)[n] for n in bodies.names]),
            np.array([S.gb_mass_fractions(sex)[n] for n in bodies.names]))
        ratio = (centre[1] - bodies.floor_cm) / bodies.height_cm * 100
        target = 100.0 - want              # 表 7 是距头顶，我们量的是距地面
        check(f"gb 站姿重心高度对上表 7 整体质心行（{label}表）",
              abs(ratio - target) <= 6.0,
              f"{ratio:.2f}% vs 国标 {target:.1f}%（表 7 给距头顶 {want}%），"
              f"差 {ratio - target:+.2f} 个百分点；抬臂 T-pose 偏高")

    gb_male = G.weighted_centre(
        np.array([S.gb_centres(joints, S.GB_MALE)[n] for n in bodies.names]),
        np.array([S.gb_mass_fractions(S.GB_MALE)[n] for n in bodies.names]))
    gb_female = G.weighted_centre(
        np.array([S.gb_centres(joints, S.GB_FEMALE)[n] for n in bodies.names]),
        np.array([S.gb_mass_fractions(S.GB_FEMALE)[n] for n in bodies.names]))
    check("gb 性别互换会改变结果", np.linalg.norm(gb_male - gb_female) > 0.5,
          f"男/女表差 {np.linalg.norm(gb_male - gb_female):.3f} cm")

    # 表 2/3 的绝对值反解全躯干质心比例：段长 = m.c ÷ Lcs，比例 = 全躯干 m.c ÷ 段长。
    # 这条与 `GB.trunk_merged_ratio()` 零共享代码——上面的数字是从标准表 2/3 手抄的，
    # 那个函数走的是 `GB.TRUNK_MC_MM` + `GB.CENTROID_PCT`。守的是「合并躯干时随手取
    # 上躯干比例」那个错误（错用会偏 10 个百分点，对全身重心 2.8 cm）。
    for sex, label, up_mc, up_pct, lo_mc, lo_pct, whole_mc in (
            (S.GB_MALE, "男", 115.6, 53.6, 177.8, 40.3, 286.2),
            (S.GB_FEMALE, "女", 107.4, 49.3, 178.7, 44.6, 281.4)):
        span = up_mc / (up_pct / 100) + lo_mc / (lo_pct / 100)
        derived = whole_mc / span * 100
        used = GB.trunk_merged_ratio(sex) * 100
        check(f"合并躯干质心比例与表 2/3 反解一致（{label}表）",
              abs(derived - used) < 0.05,
              f"反解 {derived:.2f}% vs common.gb17245 {used:.2f}%；上躯干值是 "
              f"{up_pct}%，错用它会偏 {abs(up_pct - derived):.1f} 个百分点")

    # 推导点必须始终贴在所附骨上。这条抓到过真实的 bug：早先 ear_canal 每帧用世界 Y
    # 沿头颈轴插值现算，而 485 帧里有 225 帧人是倒着的、16 帧头颈轴接近水平，插值
    # 系数过零时爆掉——f91 的耳道跑到 X=5075 cm，把 anthro 那一帧整个带偏 4.2 m。
    # 绑定姿态自检、质量比闭合、重力拟合**全都过得去**，只有逐帧看幅值才抓得住。
    far, far_frame, far_name = 0.0, None, None
    for frame in range(sampler.first, sampler.last + 1):
        joints = bodies.world_joints(sampler.stamp(frame))
        head = joints["Head_M"]
        for name in S.DERIVED:
            distance = float(np.linalg.norm(joints[name] - head))
            if distance > far:
                far, far_frame, far_name = distance, frame, name
    check("推导点始终贴在所附骨上", far < 30.0,
          f"{far_name} 距 Head_M 最远 {far:.2f} cm（f{far_frame}）")


def verify_retarget(src, path, target, contacts):
    """反算产物的校验：只读产物 FBX，不信脚本内部状态。"""
    print(f"\n反算产物（{path.name}，盯 {target}）：")
    takes = {}
    for label, source in (("原始", src), ("修正", path)):
        manager, scene = F.load(source)
        try:
            sampler = Cv.Sampler(scene, (target, "mesh"), need_bed=False)
            joints = [F.find_node(scene.GetRootNode(), name) for name in
                      ("Hip_L", "Knee_L", "Ankle_L", "Chest_M", "Shoulder_R",
                       "Elbow_R")]
            data = sampler.run(wanted=(target, "mesh"),
                               extras=("volume", "foot"))
            rotation = np.array([
                [list(joint.EvaluateLocalRotation(sampler.stamp(frame)))[:3]
                 for joint in joints]
                for frame in data["frames"]])
            takes[label] = {**data, "rotation": rotation, "fps": sampler.fps}
        finally:
            manager.Destroy()

    old, new = takes["原始"], takes["修正"]
    frames = new["frames"]
    dt = 1.0 / new["fps"]
    # 腾空窗口带上首末段：这个文件开头 11 帧就是从空中开始的，用不带边界的窗口会让
    # 那段完全没被检查。
    windows = Sig.flights_from_contacts(contacts, True, frames)
    windows = [(a, b) for a, b in windows
               if int(np.searchsorted(frames, b))
               - int(np.searchsorted(frames, a)) + 1 >= 5]
    report = Sig.gravity_over_flights(new[target], frames, windows, dt,
                                      unit=Cv.CM_PER_M)
    worst = max(abs(g + 9.81) for g in report["per_flight"])
    check(f"{target} 在腾空段是精确抛物线",
          worst < 1e-4 and report["rms_max"] < 1e-3,
          f"g = {report['gravity_mean']:+.6f} ± {report['gravity_std']:.1e} m/s²，"
          f"残差 ≤ {report['rms_max']:.1e} cm（{len(windows)} 段，含首末段）")

    touching = np.zeros(len(frames), bool)
    for start, stop in contacts:
        touching[(frames >= start) & (frames <= stop)] = True
    com_gap = np.abs(new["mesh"][touching] - old["mesh"][touching]).max()
    foot_gap = np.abs(new["foot"][touching] - old["foot"][touching]).max()
    check("触床帧逐位不变", com_gap < 1e-9 and foot_gap < 1e-9,
          f"质心差 {com_gap:.1e} cm，脚底差 {foot_gap:.1e} cm")

    volume_gap = float(np.abs(new["volume"] - old["volume"]).max())
    check("姿态未被改动（体积逐帧一致）", volume_gap < 1e-6,
          f"体积最大差 {volume_gap:.2e} cm³，共 {len(frames)} 帧"
          f"（平移是刚体变换，体积必须严格不变）")
    rotation_gap = float(np.abs(new["rotation"] - old["rotation"]).max())
    check("关节旋转逐位不变", rotation_gap < 1e-9,
          f"6 个抽样关节的局部旋转最大差 {rotation_gap:.1e} 度（只动了根的平移）")
    check("脚没穿地", new["foot"].min() >= 0.0,
          f"全片脚底最低 {new['foot'].min():.2f} cm")
    print(f"         mesh 质心最高点 {old['mesh'][:, 1].max() / 100:.2f} -> "
          f"{new['mesh'][:, 1].max() / 100:.2f} m")
    return new


def verify_common_mode(results):
    """before 基线与各 after 的顶点必须几乎一致——"共模已抵消"的可执行形式。

    两端钉住 + 精确 g 之后，抛物线顶点由腾空时间 T 唯一决定（两端等高时升高 gT²/8）。
    四个产物共用同一组 T 和同一个 g，所以顶点应当只差几厘米。这条守的是一个具体的
    错误：早先拿**未修正的原动画**当 before，量出 +75 cm，看着像"改盯质心"的巨大
    效果，其实绝大部分是 T 的共模项。
    """
    print("\nbefore/after 共模检查：")
    peaks = {name: data["mesh"][:, 1].max() / 100.0
             for name, data in results.items()}
    spread = max(peaks.values()) - min(peaks.values())
    check("各产物的质心顶点几乎一致（共模已抵消）", spread < 0.40,
          "，".join(f"{k} {v:.2f} m" for k, v in peaks.items())
          + f"；极差 {spread * 100:.0f} cm")


def run(args):
    from . import __main__ as entry

    PASSED.clear()
    verify_geometry()

    src = P.find_tramp_fbx(args.src)
    manager, scene = F.load(src)
    try:
        sampler = Cv.Sampler(scene, Cv.MODES, args.body, args.bed, args.root_bone)
        print(f"\n{src.name}：{sampler.last - sampler.first + 1} 帧 @ "
              f"{sampler.fps:g} fps")
        verify_mesh(sampler, args.check_frames)
        verify_bodies(sampler)
    finally:
        manager.Destroy()

    if entry.CURVES_JSON.is_file():
        contacts = P.read_json(entry.CURVES_JSON)["contacts"]
        results = {}
        for path in sorted(entry.RETARGET_DIR.glob("*.fbx")):
            results[path.stem] = verify_retarget(src, path, path.stem, contacts)
        if len(results) > 1:
            verify_common_mode(results)
    else:
        print(f"\n（跳过反算产物校验：没有 {P.display(entry.CURVES_JSON)}）")

    print(f"\n{sum(PASSED)}/{len(PASSED)} 通过")
    if not all(PASSED):
        raise SystemExit(1)
