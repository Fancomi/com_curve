"""FBX 场景读取与逐帧求值。**整个项目只有这个模块 import fbx**。

其余模块拿到的一律是 numpy 数组，所以没有 `fbx` 轮子的机器仍然能读 CSV、画图、
跑 dive 那条线。这条约束抄自 swim_fbx_demo 的 `python/fbx_tools/scene.py`。

矩阵约定：FbxAMatrix 是**行向量**约定，`m.Get(3, 0..2)` 是平移。所以变换一批点写作
`P @ M[:3,:3] + M[3,:3]`，不是 `M @ p`。这一条搞反不会报错，只会让整个人偏到别处，
所以 `matrix_np` 的返回值一律配 `transform_points` 使用。
"""
from pathlib import Path

import fbx
import numpy as np

TYPE_NAMES = {getattr(fbx.FbxNodeAttribute.EType, name): name
              for name in dir(fbx.FbxNodeAttribute.EType) if name.startswith("e")}


class FbxError(RuntimeError):
    """场景打不开，或者里面没有需要的东西。"""


def open_manager():
    """一个挂好 IO settings 的 manager。"""
    manager = fbx.FbxManager.Create()
    if not manager:
        raise FbxError("创建 FBX SDK manager 失败")
    manager.SetIOSettings(fbx.FbxIOSettings.Create(manager, fbx.IOSROOT))
    return manager


def load(path):
    """导入 `path`，返回 (manager, scene)；调用方负责 manager.Destroy()。

    manager 拥有全部节点，必须活得比遍历久——这就是它被返回而不是内部销毁的原因。
    """
    path = Path(path)
    if not path.is_file():
        raise FbxError(f"FBX 不存在：{path}")
    manager = open_manager()
    scene = fbx.FbxScene.Create(manager, "")
    importer = fbx.FbxImporter.Create(manager, "")
    if not importer.Initialize(str(path), -1, manager.GetIOSettings()):
        message = importer.GetStatus().GetErrorString()
        importer.Destroy()
        manager.Destroy()
        raise FbxError(f"打不开 {path}：{message}")
    ok = importer.Import(scene)
    importer.Destroy()
    if not ok:
        manager.Destroy()
        raise FbxError(f"导入失败：{path}")
    return manager, scene


def save(manager, scene, path):
    """把 `scene` 以二进制原生格式写到 `path`。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exporter = fbx.FbxExporter.Create(manager, "")
    settings = manager.GetIOSettings()
    for prop in (fbx.EXP_FBX_MATERIAL, fbx.EXP_FBX_TEXTURE, fbx.EXP_FBX_SHAPE,
                 fbx.EXP_FBX_GOBO, fbx.EXP_FBX_ANIMATION,
                 fbx.EXP_FBX_GLOBAL_SETTINGS):
        settings.SetBoolProp(prop, True)
    settings.SetBoolProp(fbx.EXP_FBX_EMBEDDED, False)
    fmt = manager.GetIOPluginRegistry().GetNativeWriterFormat()
    ok = exporter.Initialize(str(path), fmt, settings)
    if ok:
        ok = exporter.Export(scene)
    exporter.Destroy()
    if not ok:
        raise FbxError(f"保存失败：{path}")
    return path


def attr_type(node):
    """节点属性的枚举名，没有属性时返回 "-"。"""
    attribute = node.GetNodeAttribute()
    if attribute is None:
        return "-"
    return TYPE_NAMES.get(attribute.GetAttributeType(),
                          str(attribute.GetAttributeType()))


def walk_nodes(root):
    """root 的全部后代，深度优先，不含 root 自己。"""
    found = []

    def visit(node):
        for index in range(node.GetChildCount()):
            child = node.GetChild(index)
            found.append(child)
            visit(child)

    visit(root)
    return found


def find_node(root, name):
    """按名字精确找一个节点。"""
    if root.GetName() == name:
        return root
    for node in walk_nodes(root):
        if node.GetName() == name:
            return node
    raise FbxError(f"没有名叫 {name!r} 的节点")


def has_skin(node):
    """节点是不是带 eSkin 变形器的 mesh。蹦床、地板这些道具没有蒙皮。"""
    if attr_type(node) != "eMesh":
        return False
    return bool(node.GetMesh().GetDeformerCount(fbx.FbxDeformer.EDeformerType.eSkin))


def skinned_mesh_nodes(root):
    """带蒙皮的 mesh 节点。"""
    return [node for node in walk_nodes(root) if has_skin(node)]


def matrix_np(matrix):
    """FbxAMatrix -> 4x4 ndarray，保持 SDK 的行向量约定。"""
    return np.array([[matrix.Get(row, col) for col in range(4)] for row in range(4)],
                    dtype=np.float64)


def transform_points(points, matrix):
    """一批点乘一个 `matrix_np` 结果。

    Apple Accelerate 的 BLAS 在 N 超过约 3000 行时会对完全正常的 matmul 报
    divide-by-zero / overflow——实测输出逐位正确且有限，只是它内部探测代码路径时
    碰了浮点标志。这里显式屏蔽，不然每帧都会刷一屏假警告。
    """
    with np.errstate(all="ignore"):
        return points @ matrix[:3, :3] + matrix[3, :3]


def geometric_matrix(node):
    """节点的 geometric transform。

    EvaluateGlobalTransform 不含它——它只作用于 mesh、不被子节点继承——但 Maya 把它
    烘进了美术看到的位置。漏掉它会让整块 mesh 跑偏。
    """
    pivot = fbx.FbxNode.EPivotSet.eSourcePivot
    matrix = fbx.FbxAMatrix()
    matrix.SetTRS(node.GetGeometricTranslation(pivot),
                  node.GetGeometricRotation(pivot),
                  node.GetGeometricScaling(pivot))
    return matrix


def control_points(geometry):
    """mesh 或 shape 的控制点，(N, 3) ndarray，局部坐标。"""
    count = geometry.GetControlPointsCount()
    out = np.empty((count, 3), dtype=np.float64)
    for index in range(count):
        point = geometry.GetControlPointAt(index)
        out[index] = (point[0], point[1], point[2])
    return out


def triangles(mesh):
    """(M, 3) 控制点索引。多边形扇形三角化。

    这个文件里 5864 个四边形 + 18 个三角形。只取前三个角会静默丢掉每个四边形的一半，
    体积会直接错一倍——所以扇形展开不是防御性代码，是必需的。
    """
    faces = []
    for polygon in range(mesh.GetPolygonCount()):
        size = mesh.GetPolygonSize(polygon)
        corners = [mesh.GetPolygonVertex(polygon, c) for c in range(size)]
        for corner in range(1, size - 1):
            faces.append((corners[0], corners[corner], corners[corner + 1]))
    return np.array(faces, dtype=np.int64)


def timing(scene):
    """(time_mode, fps, first_frame, last_frame)，取自 anim stack。

    timeline default span 在这个文件里恰好等于 stack 区间，但 stack 才是权威——
    没有 stack 的场景根本没有动画可采。副作用：把该 stack 设为 current，
    后面所有 Evaluate* 都依赖这一步。
    """
    mode = scene.GetGlobalSettings().GetTimeMode()
    fps = fbx.FbxTime.GetFrameRate(mode)
    criteria = fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId)
    if scene.GetSrcObjectCount(criteria) == 0:
        raise FbxError("场景里没有 animation stack")
    stack = scene.GetSrcObject(criteria, 0)
    scene.SetCurrentAnimationStack(stack)
    span = stack.GetLocalTimeSpan()
    return (mode, fps,
            int(span.GetStart().GetFrameCount(mode)),
            int(span.GetStop().GetFrameCount(mode)))


def base_layer(scene):
    """当前 stack 的第 0 层。BlendShape 曲线挂在这上面。"""
    stack = scene.GetCurrentAnimationStack()
    if stack is None or stack.GetMemberCount() == 0:
        raise FbxError("当前 animation stack 没有 layer")
    return stack.GetMember(0)


def time_at(frame, mode):
    """第 `frame` 帧对应的 FbxTime。"""
    stamp = fbx.FbxTime()
    stamp.SetFrame(int(frame), mode)
    return stamp


class BlendShapeStack:
    """一个 mesh 上被动画的 BlendShape 通道，按帧算出控制点位移。

    只保留有曲线的通道：这个文件里 34 条通道只有 6 条（肘部矫正）带关键帧，其余是
    姿态库里的备用形，权重恒为 0，算它们纯属浪费。
    """

    def __init__(self, mesh, layer, base):
        kind = fbx.FbxDeformer.EDeformerType.eBlendShape
        self.channels = []
        for index in range(mesh.GetDeformerCount(kind)):
            deformer = mesh.GetDeformer(index, kind)
            for slot in range(deformer.GetBlendShapeChannelCount()):
                curve = mesh.GetShapeChannel(index, slot, layer, False)
                if curve is None or curve.KeyGetCount() == 0:
                    continue
                channel = deformer.GetBlendShapeChannel(slot)
                if channel.GetTargetShapeCount() != 1:
                    # 多目标（渐进变形）要按 full weight 区间插值，这个文件里没有，
                    # 真碰到了宁可报错也不要静默按单目标算。
                    raise FbxError(f"{channel.GetName()}："
                                   f"{channel.GetTargetShapeCount()} 个目标形，不止一个")
                shape = channel.GetTargetShape(0)
                self.channels.append((channel.GetName(), curve,
                                      control_points(shape) - base))

    def names(self):
        return [name for name, _curve, _delta in self.channels]

    def weights(self, stamp):
        """各通道当前权重（0..1）。曲线值是百分比。"""
        return [curve.Evaluate(stamp)[0] / 100.0 for _n, curve, _d in self.channels]

    def apply(self, base, stamp):
        """`base` 加上当前帧的全部形变位移。"""
        out = base.copy()
        for (_name, curve, delta) in self.channels:
            weight = curve.Evaluate(stamp)[0] / 100.0
            if weight:
                out += delta * weight
        return out


class SkinnedMesh:
    """一个蒙皮 mesh，按帧给出世界坐标顶点。

    形变顺序是 BlendShape 先、蒙皮后：矫正形是在绑定姿态的局部空间里做的，
    倒过来做等于把位移当成了世界空间的偏移。
    """

    def __init__(self, node, layer):
        mesh = node.GetMesh()
        skin_kind = fbx.FbxDeformer.EDeformerType.eSkin
        if mesh.GetDeformerCount(skin_kind) != 1:
            raise FbxError(f"{node.GetName()}：期望恰好一个 skin")
        skin = mesh.GetDeformer(0, skin_kind)
        if skin.GetSkinningType() not in (fbx.FbxSkin.EType.eLinear,
                                          fbx.FbxSkin.EType.eRigid):
            raise FbxError(f"{node.GetName()}：不支持的蒙皮类型 "
                           f"{skin.GetSkinningType()}")

        self.name = node.GetName()
        self.base = control_points(mesh)
        self.triangles = triangles(mesh)
        self.shapes = BlendShapeStack(mesh, layer, self.base)

        geometric = geometric_matrix(node)
        self.clusters = []
        self.normalized = True
        for index in range(skin.GetClusterCount()):
            cluster = skin.GetCluster(index)
            if cluster.GetControlPointIndicesCount() == 0:
                # 末端骨（ToesEnd、Finger4、Eye）连了 cluster 但没有权重。
                continue
            if cluster.GetLinkMode() != fbx.FbxCluster.ELinkMode.eNormalize:
                self.normalized = False
            reference = fbx.FbxAMatrix()
            cluster.GetTransformMatrix(reference)
            link = fbx.FbxAMatrix()
            cluster.GetTransformLinkMatrix(link)
            # 绑定逆：把顶点从 mesh 局部空间搬到该骨骼的骨骼空间。
            bind = link.Inverse() * (reference * geometric)
            self.clusters.append((
                cluster.GetLink(),
                bind,
                np.array(cluster.GetControlPointIndices(), dtype=np.int64),
                np.array(cluster.GetControlPointWeights(), dtype=np.float64),
            ))
        if not self.clusters:
            raise FbxError(f"{node.GetName()}：skin 没有任何带权重的 cluster")

    def weight_sums(self):
        """每个控制点的权重之和。eNormalize 下应当恒为 1（未被任何骨影响的点为 0）。"""
        total = np.zeros(len(self.base))
        for _link, _bind, indices, weights in self.clusters:
            total[indices] += weights
        return total
    def vertices(self, stamp):
        """当前帧的世界坐标顶点，(N, 3)。"""
        local = self.shapes.apply(self.base, stamp)
        out = np.zeros_like(local)
        for link, bind, indices, weights in self.clusters:
            matrix = matrix_np(link.EvaluateGlobalTransform(stamp) * bind)
            out[indices] += transform_points(local[indices], matrix) * weights[:, None]
        return out


def node_translation(node, stamp):
    """节点世界位置，(3,) ndarray。"""
    point = node.EvaluateGlobalTransform(stamp).GetT()
    return np.array([point[0], point[1], point[2]], dtype=np.float64)


class DeformedMesh:
    """没有蒙皮、只有 BlendShape + 节点变换的 mesh。蹦床床面就是这种。"""

    def __init__(self, node, layer):
        mesh = node.GetMesh()
        self.node = node
        self.name = node.GetName()
        self.base = control_points(mesh)
        self.triangles = triangles(mesh)
        self.shapes = BlendShapeStack(mesh, layer, self.base)
        self.geometric = geometric_matrix(node)

    def vertices(self, stamp):
        local = self.shapes.apply(self.base, stamp)
        matrix = matrix_np(self.node.EvaluateGlobalTransform(stamp) * self.geometric)
        return transform_points(local, matrix)

    def deform_amount(self, stamp):
        """全部形变权重之和。>0 即床面正在被压——这就是触床信号。"""
        return float(sum(self.shapes.weights(stamp)))
