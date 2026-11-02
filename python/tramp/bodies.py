"""14 段刚体：编译期定质量、冻结质心与惯量，运行时只做 FK。

两个用途共用这一套骨架：

  gb      重心口径。质量与质心比例来自国标 GB/T 17245-2004（表 6/7），男/女两套值，
          段质心按表 7 落在关节连线上，与网格无关。
  spin    整体旋转修复。要的是每段的**质量分布**（角动量需要惯量），国标只给质量比
          与质心、不给惯量，所以用绑定姿态的实体体素按段切开，每段 m = ρV（ρ = 1000），
          段质心与惯量冻结在绑定姿态、逐帧刚体变换。
"""
import numpy as np

from ..common import geometry as G
from . import scene as F
from . import segments as S

# 体素边长（cm）。1.2 cm 下 36571 个体素、总体积误差 +0.26%；再细一倍收益很小
# （2.0 cm 已经是 +0.54%），而这是一次性的编译期开销，不影响逐帧速度。
VOXEL_PITCH = 1.2


def bind_pose(node, skin):
    """(绑定姿态世界顶点, 各骨绑定世界位置)。

    `TransformLinkMatrix` 是骨骼在绑定时刻的世界矩阵，`TransformMatrix` 是网格节点
    在同一时刻的世界矩阵——两者取自同一个 cluster，天然同一个姿态，不需要去动画里
    找哪一帧是绑定姿态（这个文件里根本没有那一帧）。
    """
    import fbx

    mesh = node.GetMesh()
    deformer = mesh.GetDeformer(0, fbx.FbxDeformer.EDeformerType.eSkin)
    reference = fbx.FbxAMatrix()
    deformer.GetCluster(0).GetTransformMatrix(reference)
    vertices = F.transform_points(
        skin.base, F.matrix_np(reference * F.geometric_matrix(node)))

    joints = {}
    for index in range(deformer.GetClusterCount()):
        cluster = deformer.GetCluster(index)
        link = fbx.FbxAMatrix()
        cluster.GetTransformLinkMatrix(link)
        point = link.GetT()
        joints[cluster.GetLink().GetName()] = np.array(
            [point[0], point[1], point[2]], dtype=np.float64)
    return vertices, joints


def bind_matrix_inverse(node, driver):
    """驱动骨绑定世界矩阵的逆（4×4，行向量约定）。"""
    import fbx

    deformer = node.GetMesh().GetDeformer(0, fbx.FbxDeformer.EDeformerType.eSkin)
    for index in range(deformer.GetClusterCount()):
        cluster = deformer.GetCluster(index)
        if cluster.GetLink().GetName() != driver:
            continue
        link = fbx.FbxAMatrix()
        cluster.GetTransformLinkMatrix(link)
        return F.matrix_np(link.Inverse())
    raise ValueError(f"驱动骨 {driver} 没有 skin cluster，拿不到绑定矩阵")


def segment_weights(skin):
    """(14, 顶点数)：把 71 根骨的蒙皮权重按体段汇总。"""
    index = {name: i for i, name in enumerate(S.NAMES)}
    matrix = np.zeros((len(S.NAMES), len(skin.base)))
    for link, _bind, indices, weights in skin.clusters:
        bone = link.GetName()
        segment = S.bone_to_segment(bone)
        if segment is None:
            raise ValueError(f"骨 {bone} 没有对应的体段")
        matrix[index[segment], indices] += weights
    return matrix


def assign_by_skin(points, vertices, weights, neighbours=6):
    """体素 -> 体段。取最近 `neighbours` 个表面顶点的段权重之和最大的那一段。

    比骨线距离准：蒙皮权重是这个绑定对"哪块肉跟着哪根骨"的权威答案，几何距离只是
    近似。实测按权重划分头颈段拿到 8.06% 体积（Dempster 表 8.10%），按骨线距离是
    11.27%——下颌与颈周围的肉被判给了头颈中轴线而本该属于躯干。14 段的 L1 偏差
    26.1 vs 34.8 个百分点。

    取 6 个邻居而不是 1 个：单个最近顶点在腋下、裆部这类交界处会跳变，那里表面顶点
    的段属性本身就是混的，多取几个再按权重投票要稳。
    """
    from scipy.spatial import cKDTree

    _distance, nearest = cKDTree(vertices).query(points, k=neighbours)
    return weights[:, nearest].sum(2).argmax(0)


def assign_by_bone_lines(points, joints):
    """体素 -> 体段，按骨线最近距离。`tramp.verify` 用它做独立对照。"""
    distances = np.empty((len(S.NAMES), len(points)))
    for index, name in enumerate(S.NAMES):
        best = np.full(len(points), np.inf)
        for near_name, far_name in S.SEGMENT_LINES[name]:
            near = joints[near_name]
            direction = joints[far_name] - near
            offset = points - near
            with np.errstate(all="ignore"):
                t = np.clip(offset @ direction / (direction @ direction), 0.0, 1.0)
            best = np.minimum(
                best, np.linalg.norm(offset - t[:, None] * direction, axis=1))
        distances[index] = best
    return distances.argmin(0)


class RigidBodies:
    """绑定姿态编译出的 14 段刚体。逐帧只需 `transforms` / `world_joints`。"""

    def __init__(self, scene, node, skin, pitch=VOXEL_PITCH,
                 density=S.DENSITY, sex=S.GB_MALE):
        root = scene.GetRootNode()
        vertices, joints = bind_pose(node, skin)
        self.names = list(S.NAMES)
        self.pitch, self.density = pitch, density
        self.sex = sex
        self.height_cm = float(vertices[:, 1].max() - vertices[:, 1].min())
        self.floor_cm = float(vertices[:, 1].min())
        self.bind_joints = joints
        self.bind_vertices = vertices

        voxels = G.interior_voxels(vertices, skin.triangles, pitch)
        which = assign_by_skin(voxels, vertices, segment_weights(skin))
        cell = pitch ** 3
        self.voxel_count = len(voxels)
        self.voxel_volume_cm3 = float(len(voxels) * cell)

        self.volume_cm3 = np.empty(len(self.names))
        # 冻结的段质心与惯量存在**绑定姿态的世界系**里，配合 bind_inverse 一起用。
        self.bind_centre = np.empty((len(self.names), 3))
        self.bind_inertia = np.empty((len(self.names), 3, 3))   # kg·cm²，绕段质心
        for index, name in enumerate(self.names):
            picked = voxels[which == index]
            if not len(picked):
                raise ValueError(f"体段 {name} 没有分到任何体素")
            self.volume_cm3[index] = len(picked) * cell
            self.bind_centre[index] = picked.mean(0)
            offset = picked - self.bind_centre[index]
            with np.errstate(all="ignore"):
                self.bind_inertia[index] = cell * 1e-6 * density * (
                    np.eye(3) * (offset ** 2).sum() - offset.T @ offset)
        self.mass_kg = self.volume_cm3 * 1e-6 * density
        self.gb_mass = np.array([S.gb_mass_fractions(sex)[name]
                                 for name in self.names])
        self.drivers, self.bind_inverse = [], []
        for _name, _proximal, _distal, driver in S.SEGMENTS:
            self.drivers.append(F.find_node(root, driver))
            self.bind_inverse.append(bind_matrix_inverse(node, driver))

        needed = set()
        for _name, proximal, distal, _driver in S.SEGMENTS:
            needed.update(proximal, distal)
        for spec in S.DERIVED.values():
            needed.update(spec["height_from"], spec["axis"], {spec["attach"]})
        needed -= set(S.DERIVED)              # 推导点不是真节点
        self.joint_nodes = {name: F.find_node(root, name)
                            for name in sorted(needed)}

        # 推导点在绑定姿态里定一次，之后跟着 attach 骨做刚体变换。
        derived = S.derive_in_bind(joints)
        self.derived = {
            name: (derived[name], bind_matrix_inverse(node, spec["attach"]),
                   F.find_node(root, spec["attach"]))
            for name, spec in S.DERIVED.items()}

    def _rigid(self, point, bind_inverse, node, stamp):
        matrix = bind_inverse @ F.matrix_np(node.EvaluateGlobalTransform(stamp))
        return F.transform_points(np.reshape(point, (1, 3)), matrix)[0]

    def transforms(self, stamp):
        """各段「绑定世界 -> 当前世界」的刚体变换：(旋转 (14,3,3), 平移 (14,3) cm)。

        旋转按**列向量**约定（p' = R p + t）返回，从 SDK 的行向量矩阵转置而来——
        `spin` 的角动量全在 numpy 里算，统一成教科书写法免得每处都想一遍乘法顺序。
        """
        rotation = np.empty((len(self.drivers), 3, 3))
        translation = np.empty((len(self.drivers), 3))
        for i, driver in enumerate(self.drivers):
            matrix = self.bind_inverse[i] @ F.matrix_np(
                driver.EvaluateGlobalTransform(stamp))
            rotation[i] = matrix[:3, :3].T
            translation[i] = matrix[3, :3]
        return rotation, translation

    def world_joints(self, stamp):
        """各关节的世界位置，含 DERIVED 的推导点。gb 口径只需要这个。"""
        out = {name: F.node_translation(node, stamp)
               for name, node in self.joint_nodes.items()}
        for name, (point, inverse, attach) in self.derived.items():
            out[name] = self._rigid(point, inverse, attach, stamp)
        return out

    def gb_centres(self, stamp):
        """各段质心，按 GB/T 17245-2004 表 7 比例落在关节连线上，(14, 3)。"""
        centres = S.gb_centres(self.world_joints(stamp), self.sex)
        return np.array([centres[name] for name in self.names])

    def facts(self, mesh_volume_cm3):
        """写进 JSON 的口径与体段质量表。"""
        fractions = self.volume_cm3 / self.volume_cm3.sum()
        gb_table = S.gb_mass_fractions(self.sex)
        return {
            "voxel_pitch_cm": self.pitch,
            "voxel_count": self.voxel_count,
            "voxel_volume_l": round(self.voxel_volume_cm3 / 1000.0, 4),
            "mesh_volume_l": round(mesh_volume_cm3 / 1000.0, 4),
            "voxel_vs_mesh_pct": round(
                (self.voxel_volume_cm3 / mesh_volume_cm3 - 1) * 100, 3),
            "density_kg_m3": self.density,
            "total_mass_kg": round(float(self.mass_kg.sum()), 3),
            "bind_height_cm": round(self.height_cm, 2),
            "segments": [
                {"name": name,
                 "volume_l": round(float(self.volume_cm3[i] / 1000), 4),
                 "mass_kg": round(float(self.mass_kg[i]), 3),
                 "mesh_mass_pct": round(float(fractions[i] * 100), 2),
                 "gb_mass_pct": round(gb_table[name] * 100, 2)}
                for i, name in enumerate(self.names)],
        }
