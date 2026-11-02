"""把人体拆成 14 段刚体，按国标口径求全身重心；同一套刚体也给 `spin` 提供质量与惯量。

  gb     质量与质心比例来自国标 GB/T 17245-2004《成年人人体惯性参数》表 6/7（原件与
         OCR 文本在 `docs/成年人人体惯性参数/`），男/女两套值，质心落在关节连线上。
  mesh   不走这里：1 个可变形体，每帧重新积分体积（`curves.Sampler`）。

体素化出来的段质量（ρ = 1000 kg/m³ 的均质实体）不是重心口径，只给 `spin` 算角动量用：
那里要的是"这块肉有多重、怎么分布"，国标表只给质量比和质心，不给惯量。
"""
import numpy as np

from ..common import gb17245 as GB

# ---- 国标 GB/T 17245-2004 ---------------------------------------------------
# **表值本身在 `common/gb17245.py`**，这里只放「国标 9 类 -> 本骨架 14 段」的适配层。
# dive 那条线也用同一份表（但按标注的体表骨突组段），抄两遍必然会漂。

# 国标男子/女子位，转发过来免得调用方两处 import。
GB_MALE, GB_FEMALE = GB.MALE, GB.FEMALE

# 本骨架里 `SEGMENTS` 的 near 落在解剖**下端**的段：`Lcs` 是「距上端」，要用
# 1 − Lcs 才是「距 near」。
#   head_neck  near=Neck_M（下）  far=ear_canal（上）
#   trunk      near=髋中点（下）   far=肩中点（上）
# 其余 12 段（四肢）的 near 就是上端，直接用。搞反会让头颈质心跑到颈侧、
# 躯干质心掉到髋侧。
GB_NEAR_IS_LOWER = ("head_neck", "trunk")

# GB/T 17245-2004 表 6/7 的 9 类体段 -> 本骨架 14 段。这把「国标定义」翻译成「这个
# FBX 绑定里哪块肉属于哪类」，是 gb 口径唯一的适配层。国标把躯干分上下两段，
# 本骨架只有一个躯干刚体，所以质量求和、质心比例走 `GB.trunk_merged_ratio()`。
#
# 已知的定义差（都记在这里，不假装严格对齐）：
#   trunk      国标分上/下两段（颈椎点→胸下点→会阴点），本骨架是一整段刚体
#              （髋中点↔肩中点，477 mm vs 国标全躯干 657 mm）。
#   head_neck  国标是头顶点→颈椎点，本骨架是 Neck_M→ear_canal（Winter 的定义）。
#   foot       国标轴是内踝点→**足底**（竖直），本骨架是 Ankle→Toes（水平）。
#              比例 48.6% 两种轴向都接近中点，且足只占 1.48% 质量。
#   hand       国标远端是**中指指尖点**，本骨架是 MiddleFinger2（中指第二指节，
#              Winter 的定义），段长偏短约 40%。手占 0.64% 质量。
#
# 端点还有个更根本的差：国标表 1 的分界点是**体表骨性标志**，这里用的是关节中心
# （大腿因此偏短 3.3 个百分点身高、小腿偏长 2.6）。dive 那条线用手标的真骨突，
# 没有这笔账——两条 `gb` 曲线因此不是同一个量，README 里写明了。
GB_SEGMENT_MAP = {
    "head_neck": ("head_neck",),
    "trunk":     ("trunk",),
    "thigh":     ("thigh_L", "thigh_R"),
    "shank":     ("shank_L", "shank_R"),
    "foot":      ("foot_L", "foot_R"),
    "upperarm":  ("upperarm_L", "upperarm_R"),
    "forearm":   ("forearm_L", "forearm_R"),
    "hand":      ("hand_L", "hand_R"),
}

# 表 7 的「整体质心」行：全身重心距**头顶点**占身高的百分比，(男, 女)。转发用。
GB_WHOLE_BODY_FROM_VERTEX = GB.WHOLE_BODY_FROM_VERTEX

# 体素段质量的密度，kg/m³。水的密度，对人体是个好近似；只给 `spin` 的惯量用。
DENSITY = 1000.0

# 每段的 (近端关节, 远端关节, 驱动骨)。驱动骨是这段刚体挂在哪个坐标系下——
# 一段刚体只有一个 frame，所以躯干整段跟着 Root_M 走，脊椎的 Spine1/Chest 那几节
# 转动不会让躯干内部的质量重新分布。这正是刚体近似的含义。
#
# 近/远端给的是关节名的元组，多个时取几何中点。躯干的近端取左右髋中点、远端取左右
# 肩中点：Winter 的躯干定义是"大转子到肩关节"，单侧的量，用双侧中点才落在中线上。
#
# 实测这套骨架的段长与 Winter 的身高比例对得上（H=181.6 cm）：
#   大腿 45.7 / 参照 44.5，小腿 46.4 / 44.7，上臂 26.2 / 33.8，前臂 27.7 / 26.5
# 上臂偏短是因为这个绑定的 Shoulder_L 在肩峰内侧而非肩关节中心，差 7 cm；
# 它只影响上臂质心沿臂的位置（约 3 cm），对全身重心的贡献约 0.1 cm，可以忽略
# ——但要知道它在那儿。
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

# `ear_canal` 不是骨架里的节点，要推出来。头颈段沿用 Winter 的定义止于耳道，它是
# gb 头颈段的远端，直接决定头颈段 8% 多的质量放在哪，不能含糊。
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
    """段名去掉 `_L` / `_R` 后缀，用来查国标表。"""
    return name[:-2] if name.endswith(("_L", "_R")) else name


def gb_mass_fractions(sex=GB_MALE):
    """段名 -> 质量占全身比，按 GB/T 17245-2004 表 6（表值在 `common.gb17245`）。

    国标 9 类里躯干是上下两段，本骨架只有一个躯干刚体，所以先按 `GB_SEGMENT_MAP`
    展开到 14 段、躯干取两子段之和。闭合校验也在这里：加总必须约等于 1，抄错一个
    数字会立刻炸。容差 `GB.MASS_TOLERANCE`——国标自身舍入到 0.01%。
    """
    out = {}
    for kind, targets in GB_SEGMENT_MAP.items():
        if kind == "trunk":
            fraction = (GB.MASS_PCT["trunk_upper"][sex]
                        + GB.MASS_PCT["trunk_lower"][sex]) / 100.0
        else:
            fraction = GB.MASS_PCT[kind][sex] / 100.0
        for target in targets:
            out[target] = fraction
    total = sum(out.values())
    if abs(total - 1.0) > GB.MASS_TOLERANCE:
        raise ValueError(
            f"国标质量比之和是 {total}，与 1 差 {abs(total - 1.0):.2e}，"
            f"超出容差 {GB.MASS_TOLERANCE}（表 6 抄错了？）")
    return out


def gb_centroid_ratios(sex=GB_MALE):
    """段名 -> 质心距**本骨架 near 端**占段长的比例，按表 7。

    表 7 的 `Lcs` 是「距解剖上端」。四肢的 near 就是上端，直接用；
    `GB_NEAR_IS_LOWER` 里那两段（head_neck、trunk）的 near 在下端，要取 1 − Lcs。
    躯干是合并段，比例走 `GB.trunk_merged_ratio()`（从表 2/3 反解）。
    """
    out = {}
    for name in NAMES:
        kind = base_kind(name)
        ratio = (GB.trunk_merged_ratio(sex) if kind == "trunk"
                 else GB.CENTROID_PCT[kind][sex] / 100.0)
        out[name] = (1.0 - ratio) if kind in GB_NEAR_IS_LOWER else ratio
    return out


def gb_centres(joints, sex=GB_MALE):
    """按国标比例把每段质心放在关节连线上。返回 (段名 -> 位置)。"""
    ratios = gb_centroid_ratios(sex)
    out = {}
    for name, proximal, distal, _driver in SEGMENTS:
        near, far = _segment_endpoints(joints, proximal, distal)
        out[name] = near + (far - near) * ratios[name]
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

