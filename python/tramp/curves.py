"""四种重心口径的逐帧采样。**这是唯一一处知道"重心怎么算"的地方。**

原先五个脚本各写了一遍同样的循环（提取、体段、反算、渲染、校验），四份两两略有
差别——加第五种口径要改五个文件。现在只需在 `Sampler._point` 里加一支。

| 口径 | 定义 | 特点 |
| --- | --- | --- |
| `root` | 绑定骨架根 `Root_M`（骨盆）的世界位置 | 动画师直接 K 的量，逐帧平滑，但它是骨盆不是重心 |
| `mesh` | 蒙皮网格的均质体积重心（散度定理） | 可变形体，每帧重新积分 |
| `mujoco` | 仿 MuJoCo `subtree_com`：14 刚体，质量 = 体素体积 × ρ | 刚体近似，质量由网格形状决定 |
| `anthro` | Dempster/Winter 体段质量比，质心在关节连线上 | 质量由人体测量学表决定，完全不看网格 |

单位是**厘米**（场景 system unit），Y 轴朝上，右手系。
"""
import numpy as np

from ..common import geometry as G
from ..common import signal as Sig
from . import scene as F
from .bodies import RigidBodies

MODES = ("root", "mesh", "mujoco", "anthro")
SEGMENTED = ("mujoco", "anthro")            # 有 14 个体段质心可画的口径
CM_PER_M = 100.0
MODE_DOC = {
    "root": "world translation of the bind-skeleton root (pelvis)",
    "mesh": "homogeneous volume centroid of the skinned mesh, per frame",
    "mujoco": "MuJoCo subtree_com style: 14 rigid bodies, mass from voxelised "
              "mesh volume x density, segment centroid frozen in the bind pose",
    "anthro": "Dempster(1955)/Winter(2009) segment mass fractions with centroids "
              "on joint lines",
}


def pick_body(scene, name=None):
    """人体网格：显式指名，或场景里唯一带蒙皮的 mesh。"""
    root = scene.GetRootNode()
    if name:
        return F.find_node(root, name)
    nodes = F.skinned_mesh_nodes(root)
    if len(nodes) != 1:
        found = [n.GetName() for n in nodes]
        raise SystemExit(f"需要 --body 指名：场景里有 {len(found)} 个蒙皮网格 {found}")
    return nodes[0]


def pick_bed(scene, layer, name=None):
    """蹦床床面：显式指名，或没有蒙皮但 BlendShape 被 K 了的那个 mesh。

    触床用床面的形变权重判定而不是脚的高度阈值：床面被压时权重才非零，这是文件里
    对"接触"最直接的记录。实测两者给出**完全相同**的 11 段区间，但权重不需要调阈值，
    而且落地缓冲与起跳蹬伸之间不会混淆。
    """
    root = scene.GetRootNode()
    if name:
        return F.DeformedMesh(F.find_node(root, name), layer)
    found = [F.DeformedMesh(node, layer) for node in F.walk_nodes(root)
             if F.attr_type(node) == "eMesh" and not F.has_skin(node)]
    found = [mesh for mesh in found if mesh.shapes.channels]
    if len(found) != 1:
        raise SystemExit(f"需要 --bed 指名：找到 {len(found)} 个候选床面 "
                         f"{[m.name for m in found]}")
    return found[0]


class Sampler:
    """一个已打开的场景 + 它能采出的所有曲线。

    `wanted` 决定建什么：只要 root/mesh 时不碰体素化（省 0.3 s 与 scipy 依赖），
    要 mujoco/anthro 时才编译 14 段刚体。
    """

    def __init__(self, scene, wanted=MODES, body=None, bed=None,
                 root_bone="Root_M", need_bed=True):
        self.scene = scene
        self.mode, self.fps, self.first, self.last = F.timing(scene)
        layer = F.base_layer(scene)
        self.body_node = pick_body(scene, body)
        self.skin = F.SkinnedMesh(self.body_node, layer)
        self.root_bone = F.find_node(scene.GetRootNode(), root_bone)
        self.root_bone_name = root_bone
        self.wanted = tuple(wanted)
        self.bed = pick_bed(scene, layer, bed) if need_bed else None
        self.bodies = (RigidBodies(scene, self.body_node, self.skin)
                       if any(k in SEGMENTED for k in self.wanted) else None)

    # ---- 单帧 ----------------------------------------------------------

    def stamp(self, frame):
        return F.time_at(frame, self.mode)

    def _point(self, key, stamp, cache):
        """一种口径在这一帧的位置。`cache` 复用同帧的顶点与体段质心。"""
        if key == "root":
            return F.node_translation(self.root_bone, stamp)
        if key == "mesh":
            return cache["mesh"]
        centres = self.segment_centres(key, stamp, cache)
        masses = (self.bodies.mass_kg if key == "mujoco"
                  else self.bodies.anthro_mass)
        return G.weighted_centre(centres, masses)

    def segment_centres(self, key, stamp, cache=None):
        """`mujoco` / `anthro` 的 14 个体段质心，(14, 3)。"""
        if cache is not None and key in cache:
            return cache[key]
        centres = (self.bodies.world_centres(stamp) if key == "mujoco"
                   else self.bodies.anthro_centres(stamp))
        if cache is not None:
            cache[key] = centres
        return centres

    # ---- 全片 ----------------------------------------------------------

    def run(self, frames=None, wanted=None, extras=(), progress=None):
        """扫全片。返回 {"frames", 各口径, 以及 extras 要的列}。

        extras 可选：
          volume    mesh 体积 cm³
          contact   床面形变权重之和（>0 即触床）
          foot      人体最低点 Y
          bed_low   床面最低点 Y
          bbox      全片人体包围盒，写成 ("low", "high") 两个 (3,)
          segments  各 SEGMENTED 口径的逐帧 14 段质心，(N, 14, 3)
        """
        keys = tuple(wanted or self.wanted)
        frames = (np.arange(self.first, self.last + 1) if frames is None
                  else np.asarray(frames))
        extras = set(extras)
        count = len(frames)
        out = {"frames": frames}
        out.update({key: np.empty((count, 3)) for key in keys})
        for name in ("volume", "contact", "foot", "bed_low"):
            if name in extras:
                out[name] = np.empty(count)
        if "segments" in extras:
            out["segments"] = {key: np.empty((count, len(self.bodies.names), 3))
                               for key in keys if key in SEGMENTED}
        low, high = np.full(3, np.inf), np.full(3, -np.inf)

        need_vertices = ("mesh" in keys or {"volume", "foot", "bbox"} & extras)
        for index, frame in enumerate(frames):
            stamp = self.stamp(frame)
            cache = {}
            if need_vertices:
                vertices = self.skin.vertices(stamp)
                volume, cache["mesh"] = G.volume_centroid(
                    vertices, self.skin.triangles)
                if "volume" in extras:
                    out["volume"][index] = volume
                if "foot" in extras:
                    out["foot"][index] = vertices[:, 1].min()
                if "bbox" in extras:
                    low = np.minimum(low, vertices.min(0))
                    high = np.maximum(high, vertices.max(0))
            for key in keys:
                out[key][index] = self._point(key, stamp, cache)
                if "segments" in extras and key in SEGMENTED:
                    out["segments"][key][index] = cache[key]
            if "contact" in extras:
                out["contact"][index] = self.bed.deform_amount(stamp)
            if "bed_low" in extras:
                out["bed_low"][index] = self.bed.vertices(stamp)[:, 1].min()
            if progress and (index + 1) % progress == 0:
                print(f"    {index + 1}/{count}")
        if "bbox" in extras:
            out["low"], out["high"] = low, high
        return out

    # ---- 派生量 --------------------------------------------------------

    def contacts(self, data):
        """触床帧区间（帧号，闭区间）。要求 `run` 时带了 `contact`。"""
        frames = data["frames"]
        touching = data["contact"] > 1e-6
        return [(int(frames[a]), int(frames[b]))
                for a, b in Sig.runs_of(touching)]

    def facts(self, data=None):
        """写进 JSON 的场景事实：网格、蒙皮、床面的可核对数字。"""
        closed, edges, max_shared = G.is_closed(self.skin.triangles)
        weights = self.skin.weight_sums()
        out = {
            "body": self.skin.name,
            "vertices": len(self.skin.base),
            "triangles": len(self.skin.triangles),
            "closed_manifold": closed,
            "edges": edges,
            "max_faces_per_edge": max_shared,
            "bones_weighted": len(self.skin.clusters),
            "weights_normalized": self.skin.normalized,
            "weight_sum_min": float(weights[weights > 1e-9].min()),
            "weight_sum_max": float(weights.max()),
            "animated_blendshapes": self.skin.shapes.names(),
        }
        if self.bed is not None:
            out["bed"] = self.bed.name
            out["bed_blendshapes"] = len(self.bed.shapes.channels)
        if self.bodies is not None:
            bind_volume, _c = G.volume_centroid(self.bodies.bind_vertices,
                                                self.skin.triangles)
            out["rigid_bodies"] = self.bodies.facts(bind_volume)
        return out


def flights(contacts, frames=None, include_edges=False):
    """触床区间 -> 腾空区间。见 `common.signal.flights_from_contacts`。"""
    return Sig.flights_from_contacts(contacts, include_edges, frames)


def gravity_report(data, keys, contacts, fps, trim=6):
    """各口径在腾空段的重力拟合。这是四条曲线唯一的共同物理判据。"""
    windows = flights(contacts)
    dt = 1.0 / fps
    return {
        "flights": windows,
        "trim_frames": trim,
        "per_curve": {
            key: Sig.gravity_over_flights(data[key], data["frames"], windows,
                                          dt, trim, CM_PER_M)
            for key in keys},
    }


def pairwise_stats(data, keys):
    """各口径两两之间的差，单位 cm。三维距离与仅 Y 分量各一套。"""
    out = {}
    for index, left in enumerate(keys):
        for right in keys[index + 1:]:
            delta = data[right] - data[left]
            distance = np.linalg.norm(delta, axis=1)
            out[f"{left}_to_{right}"] = {
                "dist_mean_cm": round(float(distance.mean()), 3),
                "dist_max_cm": round(float(distance.max()), 3),
                "dy_mean_cm": round(float(delta[:, 1].mean()), 3),
                "dy_std_cm": round(float(delta[:, 1].std()), 3),
                "dy_max_abs_cm": round(float(np.abs(delta[:, 1]).max()), 3),
            }
    return out
