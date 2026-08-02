"""蹦床场景的舞台与四格渲染。

渲染的东西除人体外还有蹦床床面（逐帧形变，是"人踩到床"的唯一视觉证据）和两件静态
道具。哪个轴是"正面"是量出来的：`Eye_L/R` 中点减 `Head_M` 的面朝向量全片指向 −X，
左右肩连线是 ±Z，所以正视相机在 −X、侧视在 +Z。第一版把两者写反了，"正视"格里人是
横躺的侧影。
"""
import numpy as np

from ..common import grid as Grid
from ..common import plotstyle as PS
from ..common import raster as R
from . import curves as Cv
from . import scene as F

# (键, 水平世界轴, 深度轴, 深度朝向)。深度朝向 +1 表示相机在该轴正方向看向负方向。
VIEWS = (("front", 2, 0, -1), ("side", 0, 2, +1))
VIEW_CN = {"front": "正视", "side": "侧视"}
# 标签分两行：正视格只有约 240 px 宽，一行放不下"正视 · Mesh 体积重心(均质)"。
MODE_CN = {"root": "根骨骼", "mesh": "Mesh 重心",
           "mujoco": "仿 MuJoCo", "anthro": "真实人体"}
MODE_SUB = {"root": "Root_M(骨盆)", "mesh": "均质体积重心",
            "mujoco": "14 刚体 · 网格质量", "anthro": "14 刚体 · Dempster 质量"}

BODY_TINT = np.array([176, 190, 226], dtype=np.float64)     # BGR
PROP_TINT = np.array([104, 104, 108], dtype=np.float64)
BED_TINT = np.array([132, 122, 96], dtype=np.float64)
MARKER = {key: PS.bgr(key) for key in Cv.MODES}
# before 那格的主曲线：灰蓝，"从前的样子"，不抢眼。
BEFORE_TINT = (146, 127, 109)
# 副轨迹（根骨骼虚线）用中性浅灰。它在两格里都出现，所以不能借用四种口径中任何一种
# 的颜色——尤其不能用 MARKER["root"] 的亮蓝：before 格的主角恰好就是"根骨骼被拉成
# 抛物线"，两条蓝线叠在一起分不出主次。
AUX_TINT = (176, 180, 182)
# 一格宽度的下限。标题在 23 px 字号下约 200 px 宽，窄于这个数会被裁掉右半截——
# 渲染短片段调试时最容易碰到，因为世界跨度是从所选帧算的。
MIN_WIDTH = 232
# 45 帧（1.8 s）≈ 一个完整腾空周期：再长就把上一次弹跳的弧线也拖进来。
TRAIL_SPAN = 45
# 满刻度取"一段占全身一半质量"：躯干在两种口径下都接近这个值，于是最大的圆在两格
# 都是约 19 px，不会一格大一格小地误导。各自归一会把"躯干 38% vs 50%"这个分歧擦掉。
MASS_FULL_SCALE = 0.50


class Stage:
    """场景里参与渲染的几何：人体 + 床面 + 静态道具。

    道具全是静态的（实测 `BC_Mod` 下除床面外没有任何动画曲线），所以世界顶点只算
    一次。弹簧 `pCylinder47` 有 7680 个三角形（比人体的一半还多）而在正交视图里只是
    床边缘一圈几乎看不见的细纹，放弃它换来每帧约 40% 的时间。
    """

    PROPS = ("polySurface1", "group11_pCube3")

    def __init__(self, sampler):
        root = sampler.scene.GetRootNode()
        self.sampler = sampler
        self.props = []
        for name in self.PROPS:
            node = F.find_node(root, name)
            matrix = F.matrix_np(node.EvaluateGlobalTransform()
                                 * F.geometric_matrix(node))
            mesh = node.GetMesh()
            self.props.append(
                (F.transform_points(F.control_points(mesh), matrix),
                 F.triangles(mesh)))

    def layers(self, stamp):
        """当前帧的 [(顶点, 三角形, 基色)]，按"先画远景"的顺序。

        每层单独排序而不是把所有三角形混在一起排——层与层的遮挡关系是固定的
        （道具最远、床面居中、人体最近），混排会让人体的手在某些帧被床面盖住。
        """
        out = [(v, f, PROP_TINT) for v, f in self.props]
        bed = self.sampler.bed
        out.append((bed.vertices(stamp), bed.triangles, BED_TINT))
        out.append((self.sampler.skin.vertices(stamp),
                    self.sampler.skin.triangles, BODY_TINT))
        return out

    def base(self, camera, stamp):
        """一格的底图。"""
        canvas = np.full((camera.height, camera.width, 3), R.BACKGROUND,
                         dtype=np.uint8)
        for vertices, faces, tint in self.layers(stamp):
            R.draw_mesh(canvas, camera, vertices, faces, tint)
        return canvas


def cameras(clouds, tracks, height, floor=0.0):
    """每个视角一台正交相机，共用垂直范围。"""
    ranges, y_range = Grid.frame_ranges(clouds, tracks, floor=floor,
                                        headroom=40.0)
    out = {}
    for key, axis, depth_axis, sign in VIEWS:
        # 水平留 60 cm 余量：曲线的轨迹尾会伸到人体包围盒之外，裁掉的话画面边缘
        # 会出现凭空断掉的线头。
        low, high = ranges[axis]
        out[key] = R.OrthoCamera(axis, depth_axis, sign, (low - 60.0, high + 60.0),
                                 y_range, height, MIN_WIDTH)
    return out


def views():
    return [(key, VIEW_CN[key]) for key, *_rest in VIEWS]


def metres(value):
    return f"h = {value / Cv.CM_PER_M:5.2f} m"
