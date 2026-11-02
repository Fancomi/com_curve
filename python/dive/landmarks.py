"""国标体表标志点 -> dive 的 15 段模型 -> 逐帧全身重心。

**这是 dive 线唯一知道「国标怎么分段」的地方。** 表值本身在 `common/gb17245.py`。

与 tramp 的 `gb` 口径的区别（两条线不共用代码，因为端点来源根本不同）：

  tramp   端点是绑定骨架的**关节中心**（`Hip_L`/`Knee_L`…）。国标表 1 要的是体表
          骨突，所以有系统偏差：大腿偏长 3.3 个百分点身高、小腿偏短 2.6。
  dive    端点是**手工标注的真骨突**（`inputs/gb_landmarks.json`，在 SKEL 稠密
          T-pose 上点的），端点定义与表 1 严格一致。这条更忠实。

标注存的是**顶点索引**，所以逐帧不需要任何推断：`verts[index]` 就是那个解剖点。
但 DIVF 每帧只写简化子集（skin 1379 / skel 3271），而标注在稠密网格上，所以要过
一次 `skin_vertex_mapping` / `skel_vertex_mapping` 的**逆表**。19/20 个点精确命中；
`iliospinale_R` 不在 skel 子集内，退到最近点（差 10.2 mm，记在 `facts()` 里）。
"""
import json
from pathlib import Path

import numpy as np

from ..common import gb17245 as GB

# SKEL 前向所需的两处外部依赖。稠密 T-pose 只用来把标注的**稠密索引**换成 DIVF
# 子集下标，算完就丢——逐帧计算不碰它。
HSMR_LITE = Path("/Users/penghaotian/Documents/GIT/egl_render_haotian/src")
SKEL_PKL_DIR = HSMR_LITE / "hsmr_lite" / "skel"


def dense_t_pose(hsmr_lite=None, sex="male"):
    """跑 SKEL 前向拿稠密 T-pose：(skin 6890, skel 247252)。

    `poses=0` 就是零姿态 = T-pose，与标注页面用的完全同一份几何（页面也是这么导的）。
    要 torch，所以只在启用 gb 口径时才调；`__main__._open_landmarks` 会捕获 ImportError
    并退回只采 mesh。
    """
    import sys

    import torch

    root = Path(hsmr_lite or HSMR_LITE)
    sys.path.insert(0, str(root))
    from hsmr_lite.skel_model import RenderSKEL

    model = RenderSKEL(sex, model_path=str(root / "hsmr_lite" / "skel"))
    out = model.forward(torch.zeros(1, 46), torch.zeros(1, 10),
                        torch.zeros(1, 3))
    return (out.skin_verts[0].numpy().astype(np.float64),
            out.skel_verts[0].numpy().astype(np.float64))


class Landmarks:
    """把 `gb_landmarks.json` 解析成「逐帧可取的顶点索引」。

    `resolve` 只在构造时跑一次：稠密索引 -> dive 子集下标。之后 `points(skin)`
    是纯查表 + 索引，逐帧开销可以忽略。
    """

    def __init__(self, path, skin_map, skel_map, dense_skin, dense_skel):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("skin_count") != len(dense_skin):
            raise SystemExit(
                f"{path.name} 的 skin_count 是 {raw.get('skin_count')}，"
                f"稠密网格是 {len(dense_skin)} —— 先跑 "
                f"scripts/migrate_gb_landmarks.py")
        self.source = path
        self.pose = raw.get("pose")
        self.entries = raw["landmarks"]

        # 稠密索引 -> 子集下标。DIVF 每帧只有子集，所以这一步是必须的。
        inverse = {
            "skin": {int(v): i for i, v in enumerate(skin_map)},
            "skel": {int(v): i for i, v in enumerate(skel_map)},
        }
        subset = {"skin": dense_skin[skin_map], "skel": dense_skel[skel_map]}

        self.slots = {}          # gb_id -> (层, 子集下标)
        self.fallbacks = {}      # gb_id -> 退化距离（米）
        for key, entry in self.entries.items():
            layer = entry.get("mesh", "skin")
            dense = int(entry["vertex"])
            index = inverse[layer].get(dense)
            if index is None:
                # 标的顶点被 DIVF 的抽稀丢掉了。退到子集里最近的那个点，并把距离
                # 记下来——这是个已知的、可量化的近似，不能默默吞掉。
                wanted = (dense_skin if layer == "skin" else dense_skel)[dense]
                distance = np.linalg.norm(subset[layer] - wanted, axis=1)
                index = int(distance.argmin())
                self.fallbacks[key] = float(distance[index])
            self.slots[key] = (layer, index)

    def resolve(self, key, side=None):
        """标志点 -> (层, 子集下标)。成对段传 side='R'/'L'。"""
        full = f"{key}_{side}" if side else key
        if full not in self.slots:
            raise SystemExit(f"标注里缺 {full}（{self.source.name}）")
        return self.slots[full]

    def segments(self):
        """15 段的 (段名, 上端 slot, 下端 slot)。顺序与 `GB.NAMES` 一致。"""
        out = []
        for name in GB.NAMES:
            kind = GB.base_kind(name)
            side = name[-1] if kind in GB.BILATERAL else None
            upper, lower = GB.SEGMENT_BOUNDS[kind]
            out.append((name, self.resolve(upper, side),
                        self.resolve(lower, side)))
        return out

    def facts(self):
        """写进 JSON 的标注事实。退化点必须出现在这里。"""
        return {
            "source": self.source.name,
            "pose": self.pose,
            "count": len(self.slots),
            "fallbacks": {key: round(value * 1000, 2)
                          for key, value in self.fallbacks.items()},
            "fallback_note": "标注顶点不在 DIVF 子集内，退到最近点；单位 mm",
        }


class GbSampler:
    """按国标 15 段算逐帧全身重心。质量与质心比例来自 `common.gb17245`。"""

    def __init__(self, landmarks, sex=GB.MALE):
        self.sex = sex
        self.segments = landmarks.segments()
        masses = GB.mass_fractions(sex)
        ratios = GB.centroid_ratios(sex)
        self.mass = np.array([masses[name] for name, _u, _l in self.segments])
        self.ratio = np.array([ratios[name] for name, _u, _l in self.segments])
        self.names = [name for name, _u, _l in self.segments]

    def segment_centres(self, blocks):
        """各段质心，(15, 3)。`blocks` 是 {"skin": (N,3), "skel": (M,3)}。

        表 7 的 Lcs 是「距上端」，而 `SEGMENT_BOUNDS` 也是 (上端, 下端)，
        所以直接 `upper + (lower - upper) * ratio`，没有翻转。
        """
        upper = np.array([blocks[layer][index]
                          for _n, (layer, index), _l in self.segments])
        lower = np.array([blocks[layer][index]
                          for _n, _u, (layer, index) in self.segments])
        return upper + (lower - upper) * self.ratio[:, None]

    def centre(self, blocks):
        """全身重心，(3,)。质量比之和是国标自身的 0.9999，所以要归一。"""
        centres = self.segment_centres(blocks)
        return (centres * self.mass[:, None]).sum(0) / self.mass.sum()

    def facts(self, blocks):
        """段长、占身高比例与国标反解值的对照。抓「端点接错」。

        身高取「头顶点到足底的距离」而不是 y 之差：调用方可能传 HSMR local 顶点
        （y 向下）也可能传放到平面后的 global 顶点（y 向上），取模长两边都对。
        """
        upper = np.array([blocks[layer][index]
                          for _n, (layer, index), _l in self.segments])
        lower = np.array([blocks[layer][index]
                          for _n, _u, (layer, index) in self.segments])
        lengths = np.linalg.norm(lower - upper, axis=1)
        vertex_layer, vertex_index = self.segments[0][1]
        top = blocks[vertex_layer][vertex_index]
        soles = [blocks[layer][index] for name, _u, (layer, index)
                 in self.segments if name.startswith("foot")]
        stature = float(max(np.linalg.norm(top - sole) for sole in soles))
        reference = GB.segment_length_pct(self.sex)
        return {
            "sex": "male" if self.sex == GB.MALE else "female",
            "stature_m": round(stature, 4),
            "stature_note": "头顶点到足底的距离；与 y 轴朝向无关",
            "total_mass_fraction": round(float(self.mass.sum()), 6),
            "segments": [
                {"name": name,
                 "length_m": round(float(lengths[i]), 4),
                 "pct_of_stature": round(float(lengths[i] / stature * 100), 2),
                 "gb_pct_of_stature": round(reference[GB.base_kind(name)], 2),
                 "mass_pct": round(float(self.mass[i] * 100), 2),
                 "centroid_pct_from_upper": round(float(self.ratio[i] * 100), 1)}
                for i, name in enumerate(self.names)],
        }
