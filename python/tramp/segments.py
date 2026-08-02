"""把人体拆成 14 段刚体，再按两种质量口径求全身重心。

这是 dde88bca 那次会话里定下的方案 B 与 C，两者共用同一套体段划分与前向运动学，
只在**质量从哪来**上分岔：

  mujoco —— 方案 B。质量来自网格自身：把绑定姿态的实体体积按体段切开，
            每段 m = ρV（ρ 取 MuJoCo 的默认 geom density 1000 kg/m³），
            段内质心在该段骨骼的局部系里**固定**，逐帧只做刚体变换后质量加权求和。
            这就是 MuJoCo 的 `subtree_com`：编译期从 geom 推 inertial，运行时 FK。
  anthro —— 方案 C。质量来自 Dempster(1955)/Winter(2009) 的体段质量比，与网格无关；
            段质心按表里的"距近端占段长的比例"落在关节连线上。这是运动生物力学
            对着动捕关节点算重心的经典做法。

与已有两条曲线的关系：

  root   骨盆关节，1 个点，不含任何质量概念
  mesh   1 个可变形体，每帧重新积分体积——四肢挤压会让体积重新分布
  mujoco 14 个刚体，质量由网格形状决定（均质），段内分布冻结在绑定姿态
  anthro 14 个刚体，质量由人体测量学表决定（非均质），质心在关节连线上

mujoco 与 mesh 的差就是"把可变形体冻成刚体"的代价；anthro 与 mujoco 的差就是
"真人各段密度不同"的代价。这两个差是分开的，所以四条曲线一起看才说明问题。
"""
import numpy as np

# --- Dempster(1955) / Winter(2009) 14 段模型 ---------------------------------
# (质量占全身比, 质心距近端占段长的比例)。这 14 段的质量比之和恰好是 1.000——
# 不是巧合，Winter 的表就是按这个闭合关系整理的，所以下面的断言必须成立。
# 值取自 Winter《Biomechanics and Motor Control of Human Movement》表 4.1。
DEMPSTER = {
    "trunk":     (0.4970, 0.500),   # 大转子 -> 肩关节
    "head_neck": (0.0810, 1.000),   # C7-T1 -> 耳道；比例 1.000 即质心落在耳道处
    "thigh":     (0.1000, 0.433),   # 大转子 -> 股骨髁（膝）
    "shank":     (0.0465, 0.433),   # 膝 -> 内踝
    "foot":      (0.0145, 0.500),   # 外踝 -> 第二跖骨头
    "upperarm":  (0.0280, 0.436),   # 肩关节 -> 肘
    "forearm":   (0.0160, 0.430),   # 肘 -> 尺骨茎突（腕）
    "hand":      (0.0060, 0.506),   # 腕 -> 中指第二指节
}

# MuJoCo `geom` 的默认密度，kg/m³。文档口径：不写 density 也不写 mass 时，
# 编译器按均质实体 + 这个密度从 mesh 推 inertial。水的密度，对人体是个好近似。
MUJOCO_DENSITY = 1000.0

# 每段的 (近端关节, 远端关节, 驱动骨)。驱动骨是这段刚体挂在哪个坐标系下——
# MuJoCo 里一个 body 只有一个 frame，所以躯干整段跟着 Root_M 走，脊椎的
# Spine1/Chest 那几节转动不会让躯干内部的质量重新分布。这正是刚体近似的含义。
#
# 近/远端给的是关节名的元组，多个时取几何中点。躯干的近端取左右髋中点、远端取左右
# 肩中点：Winter 的躯干定义是"大转子到肩关节"，单侧的量，用双侧中点才落在中线上。
#
# 实测这套骨架的段长与 Winter 的身高比例对得上（H=181.6 cm）：
#   大腿 45.7 / 参照 44.5，小腿 46.4 / 44.7，上臂 26.2 / 33.8，前臂 27.7 / 26.5
# 上臂偏短是因为这个绑定的 Shoulder_L 在肩峰内侧而非肩关节中心，差 7 cm；
# 它只影响上臂质心沿臂的位置（0.436 × 7 ≈ 3 cm），对全身重心的贡献
# 是 0.028 × 3 ≈ 0.1 cm，可以忽略——但要知道它在那儿。
#
# 手的远端用 MiddleFinger2（掌指关节到中指第二指节），不是 MiddleFinger4（指尖）：
# Winter 的手段定义止于"中指第二指节"，用指尖会把段长拉长 40%。
SEGMENTS = (
    ("trunk",        ("Hip_L", "Hip_R"), ("Shoulder_L", "Shoulder_R"), "Root_M"),
    ("head_neck",    ("Neck_M",),        ("ear_canal",),               "Head_M"),
    ("thigh_L",      ("Hip_L",),         ("Knee_L",),                  "Hip_L"),
    ("thigh_R",      ("Hip_R",),         ("Knee_R",),                  "Hip_R"),
    ("shank_L",      ("Knee_L",),        ("Ankle_L",),                 "Knee_L"),
    ("shank_R",      ("Knee_R",),        ("Ankle_R",),                 "Knee_R"),
    ("foot_L",       ("Ankle_L",),       ("Toes_L",),                  "Ankle_L"),
    ("foot_R",       ("Ankle_R",),       ("Toes_R",),                  "Ankle_R"),
    ("upperarm_L",   ("Shoulder_L",),    ("Elbow_L",),                 "Shoulder_L"),
    ("upperarm_R",   ("Shoulder_R",),    ("Elbow_R",),                 "Shoulder_R"),
    ("forearm_L",    ("Elbow_L",),       ("Wrist_L",),                 "Elbow_L"),
    ("forearm_R",    ("Elbow_R",),       ("Wrist_R",),                 "Elbow_R"),
    ("hand_L",       ("Wrist_L",),       ("MiddleFinger2_L",),         "Wrist_L"),
    ("hand_R",       ("Wrist_R",),       ("MiddleFinger2_R",),         "Wrist_R"),
)

# `ear_canal` 不是骨架里的节点，要推出来。Winter 的头颈段止于耳道、且质心比例是
# 1.000（质心就落在耳道处），所以这个点直接决定头颈段 8.1% 质量放在哪，不能含糊。
#
# 推法：**在绑定姿态里**定出它相对头骨的局部位置，之后逐帧只做刚体变换。绑定姿态下
# 取眼睛的高度（解剖上耳道与外眼角大致同高），落在 Neck_M→HeadEnd_M 的头颈中轴线上；
# 横向取中线，前后由中轴线给——比拿眼睛当耳道靠谱，眼睛在脸前方约 7 cm。
#
# 早先的版本每帧现算这个插值，用世界 Y 当插值参数。那在站姿下没问题，在蹦床上是灾难：
# 全片 485 帧里有 225 帧人是倒着的、16 帧头颈轴接近水平，轴的 Y 跨度过零时插值系数
# 爆掉，f91 算出的耳道跑到 X=5075 cm（正常范围 ±100）。刚体化之后姿态无关，
# 这类退化不可能再发生。
DERIVED = {"ear_canal": {"height_from": ("Eye_L", "Eye_R"),
                         "axis": ("Neck_M", "HeadEnd_M"),
                         "attach": "Head_M"}}

NAMES = tuple(name for name, *_rest in SEGMENTS)

# 体素划分时每段的"骨线"：点归给距离最近的那条线。躯干给多条，因为它是个块而不是
# 一根棍——只给 Root->Chest 一条中轴线会把肩和髋周围的肉都判给四肢。
#
# 这套已经不是默认划分了（默认走蒙皮权重，见 BONE_TO_SEGMENT），保留下来是因为
# `tramp.verify` 用它做对照：两种完全独立的划分方式若都能让段质心合成回到整网重心，
# 那这个合成恒等式就不是靠某一种划分的巧合成立的。
SEGMENT_LINES = {
    "trunk":      (("Root_M", "Spine1_M"), ("Spine1_M", "Chest_M"),
                   ("Root_M", "Hip_L"), ("Root_M", "Hip_R"),
                   ("Chest_M", "Scapula_L"), ("Chest_M", "Scapula_R")),
    "head_neck":  (("Chest_M", "Neck_M"), ("Neck_M", "Head_M"),
                   ("Head_M", "HeadEnd_M")),
    "thigh_L":    (("Hip_L", "Knee_L"),),
    "thigh_R":    (("Hip_R", "Knee_R"),),
    "shank_L":    (("Knee_L", "Ankle_L"),),
    "shank_R":    (("Knee_R", "Ankle_R"),),
    "foot_L":     (("Ankle_L", "ToesEnd_L"),),
    "foot_R":     (("Ankle_R", "ToesEnd_R"),),
    "upperarm_L": (("Shoulder_L", "Elbow_L"),),
    "upperarm_R": (("Shoulder_R", "Elbow_R"),),
    "forearm_L":  (("Elbow_L", "Wrist_L"),),
    "forearm_R":  (("Elbow_R", "Wrist_R"),),
    "hand_L":     (("Wrist_L", "MiddleFinger4_L"),),
    "hand_R":     (("Wrist_R", "MiddleFinger4_R"),),
}


def bone_to_segment(bone):
    """骨骼名 -> 体段名。不认识的骨返回 None。

    默认划分走这条路而不是骨线距离：蒙皮权重就是这个绑定对"哪块肉跟着哪根骨动"的
    权威答案，几何距离只是它的近似。实测差别不小——按蒙皮权重划分，头颈段拿到
    8.06% 的体积，Dempster 表是 8.10%；按骨线距离则是 11.27%，因为下颌与颈周围的
    肉被判给了头颈中轴线而本该属于躯干。全部 14 段的 L1 偏差 26.1 vs 34.8 个百分点。

    `*Part1/2` 是这套绑定的扭曲辅助骨（HipPart1_R 之类），跟着父关节走，所以按
    前缀归到父段即可。
    """
    if bone.endswith(("_L", "_R")):
        side = bone[-1]
        stem = bone[:-2]
        if stem.startswith("Hip"):
            return f"thigh_{side}"
        if stem.startswith("Knee"):
            return f"shank_{side}"
        if stem.startswith(("Ankle", "Toes")):
            return f"foot_{side}"
        if stem.startswith("Shoulder"):
            return f"upperarm_{side}"
        if stem.startswith("Elbow"):
            return f"forearm_{side}"
        if stem.startswith(("Wrist", "Cup")) or "Finger" in stem:
            return f"hand_{side}"
        if stem.startswith("Scapula"):
            return "trunk"           # 肩胛的肉在躯干上，不在上臂
        if stem.startswith("Eye"):
            return "head_neck"
    if bone.startswith(("Root", "Spine", "Chest")):
        return "trunk"
    if bone.startswith(("Neck", "Head", "Jaw")):
        return "head_neck"
    return None


def base_kind(name):
    """段名去掉 `_L` / `_R` 后缀，用来查 Dempster 表。"""
    return name[:-2] if name.endswith(("_L", "_R")) else name


def dempster_mass_fractions():
    """段名 -> 质量占全身比。左右各算一次，总和必须是 1（表抄错了会在这里炸）。"""
    out = {name: DEMPSTER[base_kind(name)][0] for name in NAMES}
    total = sum(out.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Dempster 质量比之和是 {total}，不是 1")
    return out


def derive_in_bind(joints):
    """在绑定姿态里定出 DERIVED 各点的世界位置。只在编译期调一次。

    绑定姿态是直立的，所以"沿中轴线插值到某个高度"这件事在这里是良定义的；
    逐帧现算就不是了，见 DERIVED 的注释。
    """
    out = {}
    for name, spec in DERIVED.items():
        left, right = spec["height_from"]
        height = (joints[left][1] + joints[right][1]) / 2.0
        low, high = joints[spec["axis"][0]], joints[spec["axis"][1]]
        span = high[1] - low[1]
        if abs(span) < 1.0:
            raise ValueError(
                f"绑定姿态下 {spec['axis'][0]}->{spec['axis'][1]} 的垂直跨度只有 "
                f"{span:.3f} cm，推不出 {name}；这个绑定不是直立姿态？")
        out[name] = low + (high - low) * ((height - low[1]) / span)
    return out


def _segment_endpoints(joints, proximal, distal):
    """一段的 (近端, 远端) 世界坐标；多关节取几何中点。"""
    near = np.mean([joints[n] for n in proximal], axis=0)
    far = np.mean([joints[n] for n in distal], axis=0)
    return near, far


def anthropometric_centres(joints):
    """按 Dempster 比例把每段质心放在关节连线上。返回 (段名 -> 位置)。

    `joints` 必须已经含 DERIVED 的点（由 `RigidBodies.world_joints` 负责补齐）。
    """
    out = {}
    for name, proximal, distal, _driver in SEGMENTS:
        near, far = _segment_endpoints(joints, proximal, distal)
        ratio = DEMPSTER[base_kind(name)][1]
        out[name] = near + (far - near) * ratio
    return out
