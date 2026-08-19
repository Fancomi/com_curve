"""GB/T 17245-2004《成年人人体惯性参数》的表值。**只有数据，不认识业务口径。**

两条线都要用：tramp 按绑定骨架的关节组段、dive 按标注的体表骨突组段，
但表 6（相对质量分布）与表 7（质心相对位置）是同一份国标数字，抄两遍必然会漂。

数值逐位取自 `docs/成年人人体惯性参数/成年人人体惯性参数.md`（MinerU OCR 提取），
样本为全国成年男子 11164 人（18–60 岁）、女子 11150 人（18–55 岁）。

**表 7 的 `Lcs` 是质心距「解剖上端」占段长的百分比**，不是「距近端」。标准正文的两个
示例把这件事写死了：头颈从**头顶**量、大腿从**髂前上棘点**量；表 7 的注也说 Lcs 指
"质心上部尺寸占本体段全长的百分比"（另一列 Lcx = 100 − Lcs，20 行全部闭合）。
别跟表 1 的"质心测量起点"混起来——那一列给表 2/3 的绝对 mm 值用，多数段取的是
**远侧**点（大腿是胫骨点、上臂是桡骨点、手是中指指尖点）。

国标表 1 的分界点全是**体表骨性标志**，不是关节中心：标准 4.2 写的是"以明显的骨性
标志为分界点"，第 5 章的测量项目全是"颈椎点高""胫骨点高"这类离地高度，样本是活人
用测高仪量的。用关节中心近似会有系统偏差（大腿 +3.3 个百分点身高、小腿 −2.6），
调用方自己记这笔账。
"""

# 表 6 相对质量分布（%），(男, 女)。9 类都是单侧值；双侧合计男 99.99%、女 100.01%。
MASS_PCT = {
    "head_neck": (8.62, 8.20),
    "trunk_upper": (16.82, 16.35),
    "trunk_lower": (27.23, 27.48),
    "thigh": (14.19, 14.10),
    "shank": (3.67, 4.43),
    "foot": (1.48, 1.24),
    "upperarm": (2.43, 2.66),
    "forearm": (1.25, 1.14),
    "hand": (0.64, 0.42),
}

# 表 6 双侧合计的容差。国标自身舍入到 0.01%，加总后男 99.99 / 女 100.01。
MASS_TOLERANCE = 5e-4

# 表 7 的 `Lcs` 列（%），(男, 女)：质心距该段**解剖上端**占段长的比例。
CENTROID_PCT = {
    "head_neck": (46.9, 47.3),      # 自头顶点
    "trunk_upper": (53.6, 49.3),    # 自颈椎点
    "trunk_lower": (40.3, 44.6),    # 自胸下点
    "thigh": (45.3, 44.2),          # 自髂前上棘点
    "shank": (39.3, 42.5),          # 自胫骨点
    "foot": (48.6, 45.1),           # 自内踝点
    "upperarm": (47.8, 46.7),       # 自肩峰点
    "forearm": (42.4, 45.3),        # 自桡骨点
    "hand": (36.6, 34.9),           # 自桡骨茎突点
}

# 表 7 的「整体质心」行：全身重心距**头顶点**占身高的百分比，(男, 女)。
# 换成「距地面」是 100 − 这个数：男 56.2%、女 55.5%。这是国标自带的、与分段实现
# 无关的校验基准，比抄文献的 55–57% 硬。
WHOLE_BODY_FROM_VERTEX = (43.8, 44.5)

# 表 2/表 3 的绝对值（mm，男/女），用来反解段长与合并躯干的比例。
# (上躯干 m.c, 下躯干 m.c, 全躯干 m.c)
TRUNK_MC_MM = {"male": (115.6, 177.8, 286.2), "female": (107.4, 178.7, 281.4)}

MALE, FEMALE = 0, 1

# 表 1 的分界点：段名 -> (上端标志, 下端标志)。标志名与 `gb_landmarks.json` 的
# `gb_key` 一致，成对段的左右由调用方加 `_R`/`_L`。
SEGMENT_BOUNDS = {
    "head_neck": ("vertex", "cervicale"),
    "trunk_upper": ("cervicale", "substernale"),
    "trunk_lower": ("substernale", "perineum"),
    "thigh": ("iliospinale", "tibiale"),
    "shank": ("tibiale", "sphyrion"),
    "foot": ("sphyrion", "sole"),
    "upperarm": ("acromion", "radiale"),
    "forearm": ("radiale", "stylion"),
    "hand": ("stylion", "dactylion"),
}

# 哪些段左右各一份。国标表 1 的注：「左右对称部分，如上臂、前臂、手、大腿、小腿、
# 足各部位，其体段分界点的名称相同」。
BILATERAL = ("thigh", "shank", "foot", "upperarm", "forearm", "hand")

# 15 段模型：4 个中线段 + 6 类 × 2 侧 = 16？不是——躯干拆上下两段，头颈 1 段，
# 于是 3 + 6×2 = 15。这与标准 4.2 的「共 15 个部位」一致。
NAMES = tuple(
    [name for name in SEGMENT_BOUNDS if name not in BILATERAL]
    + [f"{name}_{side}" for name in SEGMENT_BOUNDS if name in BILATERAL
       for side in ("R", "L")])


def base_kind(name):
    """段名去掉 `_R` / `_L` 后缀，用来查表。"""
    return name[:-2] if name.endswith(("_R", "_L")) else name


def mass_fractions(sex=MALE, names=NAMES):
    """段名 -> 质量占全身比。左右共享同一比例；总和必须约等于 1。"""
    out = {name: MASS_PCT[base_kind(name)][sex] / 100.0 for name in names}
    total = sum(out.values())
    if abs(total - 1.0) > MASS_TOLERANCE:
        raise ValueError(
            f"国标质量比之和是 {total}，与 1 差 {abs(total - 1.0):.2e}，"
            f"超出容差 {MASS_TOLERANCE}（表 6 抄错了？）")
    return out


def centroid_ratios(sex=MALE, names=NAMES):
    """段名 -> 质心距该段**上端**占段长的比例。与 `SEGMENT_BOUNDS` 的顺序一致，
    所以调用方直接 `upper + (lower - upper) * ratio` 即可。"""
    return {name: CENTROID_PCT[base_kind(name)][sex] / 100.0 for name in names}


def trunk_merged_ratio(sex=MALE):
    """把上下躯干合成一段时的质心比例（距上端，0..1）。

    表 7 只给上躯干、下躯干各自的值。要把躯干当**一整段**用（tramp 的骨架只有一个
    躯干刚体），合并比例必须自己推——**不能随手取上躯干的 53.6%**，那是上躯干自己
    215.7 mm 段长上的比例。

    从表 2/表 3 的绝对值反解（段长 = m.c ÷ Lcs）：
      男 上躯干 115.6/0.536 = 215.7 mm，下躯干 177.8/0.403 = 441.2 mm，全躯干 656.9
         全躯干 m.c 286.2 → 286.2/656.9 = 43.57%
      女 上躯干 107.4/0.493 = 217.8 mm，下躯干 178.7/0.446 = 400.7 mm，全躯干 618.5
         全躯干 m.c 281.4 → 281.4/618.5 = 45.50%
    """
    upper_mc, lower_mc, whole_mc = TRUNK_MC_MM["male" if sex == MALE else "female"][:3]
    span = (upper_mc / (CENTROID_PCT["trunk_upper"][sex] / 100.0)
            + lower_mc / (CENTROID_PCT["trunk_lower"][sex] / 100.0))
    return whole_mc / span


def segment_length_pct(sex=MALE):
    """段名 -> 段长占身高的百分比，从表 2/3 的绝对值反解。

    给校验用：标注/骨架组出来的段长若掉出这个比例太多，就是端点接错了。
    身高由表 7「整体质心」行反解：男 734.2/0.438 = 1676.3 mm。
    """
    from_vertex = WHOLE_BODY_FROM_VERTEX[sex] / 100.0
    whole_mc = 734.2 if sex == MALE else 698.7
    stature = whole_mc / from_vertex
    # 表 2/3 的 m.c 与表 1 的「质心测量起点」配套：中线三段自上端量，四肢自下端量。
    mc_mm = {
        "male": {"head_neck": 117.8, "trunk_upper": 115.6, "trunk_lower": 177.8,
                 "thigh": 254.5, "shank": 224.1, "foot": 38.2,
                 "upperarm": 163.3, "forearm": 136.6, "hand": 114.2},
        "female": {"head_neck": 111.8, "trunk_upper": 107.4, "trunk_lower": 178.7,
                   "thigh": 244.5, "shank": 197.6, "foot": 36.2,
                   "upperarm": 151.4, "forearm": 124.9, "hand": 115.0},
    }["male" if sex == MALE else "female"]
    from_upper = ("head_neck", "trunk_upper", "trunk_lower")
    out = {}
    for kind, value in mc_mm.items():
        upper = CENTROID_PCT[kind][sex] / 100.0
        ratio = upper if kind in from_upper else (1.0 - upper)
        out[kind] = value / ratio / stature * 100.0
    return out
