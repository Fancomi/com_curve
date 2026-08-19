#!/usr/bin/env python3
"""导出**稠密** T-pose 的 skin/skel 网格给 tools/gb_landmark_label/ 那个标注页面。

**顶点索引就是 SKEL/SMPL 原生索引，不做任何抽稀、不建中间映射。**
标出来的 `vertex` 直接可以索引 `skin_template_v`(6890) / `skel_template_v`(247252)，
下游拿到 SKEL 前向输出后 `verts[index]` 即可，中间没有任何需要维护的对照表。

早先那一版用 HSMR 的两个简化 OBJ（1379 / 3271），代价是：抽稀没有成对保留顶点，
1379 里只有 587 个点的镜像还在子集内（42.6%），骨盆区 156 个点里只有 80 个 ——
标左右成对的骨突时对侧压根没有对应顶点。稠密网格没这个问题。

镜像表 `twin` 一并导出，页面用它显示对侧位置：
  skin  用 pkl 自带的 `vert_sym_idxs`（是个合规对合，134 个中线点自映射）
  skel  没有自带表，按骨构造：左右同名骨的顶点**一一对应**（实测 T-pose 下逐位
        镜像偏差中位 2.4~18 mm），所以用骨内下标偏移；`hand_r` 恰好是 `hand_l`
        的 2 倍所以取模。中线骨（pelvis/lumbar/thorax/head）在骨内做几何镜像。
        这样得到的偏差中位 4.3 mm、90% 10.2 mm，比「全网格找最近」的 15.5 mm /
        最大 176 mm 好一个量级。

产物三个文件（页面按 manifest 里的偏移解析二进制）：
  outputs/gb_landmark_asset.json   元数据：骨名、标志点定义、关节、buffer 布局
  outputs/gb_landmark_skin.bin     positions f32 + faces u16 + bone u8 + twin i32
  outputs/gb_landmark_skel.bin     positions f32 + faces u32 + bone u8 + twin i32

法向不导出，页面用 `computeVertexNormals()` 现算（省 3 MB，结果一致）。
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.common import paths as P            # noqa: E402

SKEL_PKL = Path("/Users/penghaotian/Documents/GIT/egl_render_haotian/src/"
                "hsmr_lite/skel/skel_male.pkl")
HSMR_LITE = Path("/Users/penghaotian/Documents/GIT/egl_render_haotian/src")

# 国标 GB/T 17245-2004 表 1 的 12 类体表骨性标志。`side` 为 True 时左右各一个。
# `bone` 是该点所在的 SKEL 骨（词干，不含 _r/_l），页面用它做「隔离相关骨」。
LANDMARKS = [
    dict(key="vertex", cn="头顶点", side=False, bone="head",
         hint="头部最高点。直立时颅顶正中，取中线上 y 最大的那个顶点。"),
    dict(key="cervicale", cn="颈椎点", side=False, bone="thorax",
         hint="第七颈椎（C7）棘突尖。低头时后颈最突出的那块骨头，在背面中线上。"),
    dict(key="substernale", cn="胸下点", side=False, bone="thorax",
         hint="胸骨体下缘与正中矢状面的交点（标准 3.6 条）。胸前中线、"
              "胸骨最下端，约在两侧肋弓交汇处上方。"),
    dict(key="perineum", cn="会阴点", side=False, bone="pelvis",
         hint="两腿分叉处最低点，躯干与大腿的分界。取裆部中线上 y 最小的顶点。"),
    dict(key="iliospinale", cn="髂前上棘点", side=True, bone="pelvis",
         hint="髂骨前上方的骨突，腰侧最前最上那个能摸到的尖。比髋关节中心明显偏"
              "上偏前——这正是国标大腿比关节口径长 3.3% 身高的原因。"),
    dict(key="tibiale", cn="胫骨点", side=True, bone="tibia",
         hint="胫骨内侧上端的最高点（膝内侧下方）。**不是膝关节中心**，比它低"
              "几厘米，是大腿与小腿的国标分界。"),
    dict(key="sphyrion", cn="内踝点", side=True, bone="talus",
         hint="内踝（脚踝内侧那块骨头）最下端。小腿与足的分界。"),
    dict(key="sole", cn="足底", side=True, bone="calcn",
         hint="足底最低点（着地面）。取该侧脚上 y 最小的顶点。"),
    dict(key="acromion", cn="肩峰点", side=True, bone="scapula",
         hint="肩峰外侧缘最突出点，肩膀最外上方的骨突。上臂的近端。"),
    dict(key="radiale", cn="桡骨点", side=True, bone="ulna",
         hint="桡骨头上缘，肘外侧那个能摸到的骨突。上臂与前臂的分界。"),
    dict(key="stylion", cn="桡骨茎突点", side=True, bone="radius",
         hint="桡骨茎突最下端，手腕拇指侧那个骨突。前臂与手的分界。"),
    dict(key="dactylion", cn="中指指尖点", side=True, bone="hand",
         hint="中指最末端。手段的远端。"),
]


def t_pose(pkl, hsmr_lite):
    """跑 SKEL 前向拿 T-pose：(skin 6890, skel 247252, joints 24)。

    `poses=0` 就是 SKEL 的零姿态 = T-pose。走前向而不是直接读 `skel_template_v`，
    是因为后者是**骨局部坐标**（各骨绕自己原点，pelvis 的 bbox 只有 0.16 m），
    不乘 24 个骨变换拼不成人。关节也是前向白给的。
    """
    import torch

    sys.path.insert(0, str(hsmr_lite))
    from hsmr_lite.skel_model import RenderSKEL

    model = RenderSKEL("male", model_path=str(pkl.parent))
    out = model.forward(torch.zeros(1, 46), torch.zeros(1, 10),
                        torch.zeros(1, 3))
    return (out.skin_verts[0].numpy().astype(np.float64),
            out.skel_verts[0].numpy().astype(np.float64),
            out.joints[0].numpy().astype(np.float64))


def bone_labels_skin(weights, count):
    """skin 顶点 -> 骨：蒙皮权重最大的那根。"""
    rows = weights.tocsr()
    return np.asarray(rows.argmax(axis=1)).ravel().astype(np.uint8)


def bone_labels_skel(indices, count):
    """skel 顶点 -> 骨。`bone_verts_indices` 是连续区间，与 `skel_weights_rigid`
    的 argmax 逐位一致（核对过），用区间更省；区间无缝覆盖 0..247252。"""
    out = np.zeros(count, np.uint8)
    for bone, (start, stop) in enumerate(indices):
        out[start:stop] = bone
    return out


def skin_twin(sym):
    """skin 镜像表 = pkl 自带的 `vert_sym_idxs`，直接用。

    实测它是合规的对合（`sym[sym] == arange`），134 个中线点自映射，配对几何偏差
    中位 4.7 mm、最大 14.3 mm —— SMPL 模板本身只是近似对称，这是模板的性质，
    不是我们能修的。
    """
    return sym.astype(np.int32)


def skel_twin(vertices, names, indices):
    """skel 镜像表。没有自带表，按骨构造：

      成对骨   左右同名骨的顶点**一一对应**（T-pose 下逐位镜像偏差中位 2.4~18 mm），
               所以直接用骨内下标偏移。`hand_r` 恰好是 `hand_l` 的 2 倍，取模。
      中线骨   pelvis / lumbar_body / thorax / head，在**该骨内部**做几何镜像。
               限制在骨内很关键——全网格找最近会跳到别的部位去（中位 15.5 mm、
               最大 176 mm，而按骨做是 4.3 mm / 43 mm）。
    """
    from scipy.spatial import cKDTree

    out = np.arange(len(vertices), dtype=np.int32)
    for index, name in enumerate(names):
        if not name.endswith("_r"):
            continue
        other = names.index(name[:-2] + "_l")
        a0, a1 = indices[index]
        b0, b1 = indices[other]
        na, nb = a1 - a0, b1 - b0
        out[a0:a1] = b0 + (np.arange(na) % nb)
        out[b0:b1] = a0 + (np.arange(nb) % na)
    for index, name in enumerate(names):
        if name.endswith("_r") or name.endswith("_l"):
            continue
        a0, a1 = indices[index]
        block = vertices[a0:a1]
        _distance, nearest = cKDTree(block).query(block * [-1, 1, 1])
        out[a0:a1] = a0 + nearest
    return out


def pack(path, vertices, faces, bone, twin):
    """把一层网格写成一个二进制文件，返回 manifest 片段。

    面索引按顶点数选 u16/u32：skin 6890 个顶点用 u16 省一半，skel 必须 u32。
    顺序固定为 positions -> faces -> bone -> twin。

    **每块起始偏移补齐到 4 字节**：JS 的 `new Int32Array(buffer, offset, n)` 要求
    offset 是 4 的倍数，而 bone 是 u8（skin 6890 个）会让后面的 twin 落在奇数偏移上，
    直接抛 RangeError。padding 写进文件、偏移记在 manifest 里，页面照偏移切即可。
    """
    index_type = "u16" if len(vertices) <= 65535 else "u32"
    blobs = [
        ("positions", vertices.astype("<f4").ravel()),
        ("faces", faces.astype("<u2" if index_type == "u16" else "<u4").ravel()),
        ("bone", bone.astype("<u1")),
        ("twin", twin.astype("<i4")),
    ]
    offsets, cursor, chunks = {}, 0, []
    for name, blob in blobs:
        pad = (-cursor) % 4
        if pad:
            chunks.append(b"\0" * pad)
            cursor += pad
        offsets[name] = cursor
        chunks.append(blob.tobytes())
        cursor += blob.nbytes
    path.write_bytes(b"".join(chunks))
    return {
        "file": path.name,
        "count": int(len(vertices)),
        "faces": int(len(faces)),
        "index_type": index_type,
        "offsets": offsets,
        "bytes": cursor,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl", type=Path, default=SKEL_PKL)
    parser.add_argument("--hsmr-lite", type=Path, default=HSMR_LITE,
                        help="egl_render_haotian/src，里面有 hsmr_lite 包")
    parser.add_argument("--out", type=Path, default=P.OUTPUTS)
    args = parser.parse_args()

    if not args.pkl.is_file():
        raise SystemExit(f"SKEL pkl 不存在：{args.pkl}")
    data = pickle.load(open(args.pkl, "rb"))
    if data.get("version") != "1.1.1":
        raise SystemExit(f"SKEL 版本 {data.get('version')}，期望 1.1.1")

    skin, skel, joints = t_pose(args.pkl, args.hsmr_lite)
    skin_faces = np.asarray(data["skin_template_f"])
    skel_faces = np.asarray(data["skel_template_f"])
    names = list(data["bone_names"])
    indices = np.asarray(data["bone_verts_indices"])

    span = skin.max(0) - skin.min(0)
    if span[0] < 1.3:
        raise SystemExit(f"x 跨度只有 {span[0]:.3f} m，这不是 T-pose")

    args.out.mkdir(parents=True, exist_ok=True)
    skin_pack = pack(args.out / "gb_landmark_skin.bin", skin, skin_faces,
                     bone_labels_skin(data["skin_weights"], len(skin)),
                     skin_twin(data["vert_sym_idxs"]))
    skel_pack = pack(args.out / "gb_landmark_skel.bin", skel, skel_faces,
                     bone_labels_skel(indices, len(skel)),
                     skel_twin(skel, names, indices))

    payload = {
        "source": {
            "skel_pkl": str(args.pkl),
            "pose": "SKEL forward(poses=0, betas=0, trans=0) — true T-pose",
            "note": "vertex 索引就是 SKEL/SMPL 原生索引，无抽稀、无中间映射；"
                    "下游 verts[index] 直接可用",
        },
        "axes": {"up": "+Y", "front": "+Z", "right": "-X",
                 "note": "量出来的：toes 的 z 大于 pelvis 所以前方是 +Z；"
                         "femur_r 的 x 为负所以右侧是 -X"},
        "bone_names": names,
        "bone_ranges": indices.tolist(),
        "skin": skin_pack,
        "skel": skel_pack,
        "joints": np.round(joints, 5).tolist(),
        "landmarks": LANDMARKS,
    }
    manifest = args.out / "gb_landmark_asset.json"
    manifest.write_text(json.dumps(payload, ensure_ascii=False),
                        encoding="utf-8")

    print(f"wrote {P.display(manifest)} "
          f"({manifest.stat().st_size / 1e3:.1f} KB)")
    for label, block in (("skin", skin_pack), ("skel", skel_pack)):
        print(f"  {label}: {block['count']} 顶点 / {block['faces']} 面 "
              f"({block['index_type']}) -> {block['file']} "
              f"{block['bytes'] / 1e6:.2f} MB")
    print(f"  T-pose：x 跨度 {span[0]:.3f} m，y 跨度 {span[1]:.3f} m")
    inside = ((skel >= skin.min(0) - 0.05) & (skel <= skin.max(0) + 0.05)).all(1)
    joints_in = ((joints >= skin.min(0)) & (joints <= skin.max(0))).all(1)
    print(f"  骨顶点在皮肤 bbox 内 {inside.sum()}/{len(skel)}，"
          f"关节在内 {joints_in.sum()}/24")

    for label, verts, twin in (
            ("skin", skin, skin_twin(data["vert_sym_idxs"])),
            ("skel", skel, skel_twin(skel, names, indices))):
        error = np.linalg.norm(verts[twin] * [-1, 1, 1] - verts, axis=1)
        print(f"  {label} 镜像表：中位 {np.median(error) * 1000:.2f} mm，"
              f"90% {np.percentile(error, 90) * 1000:.2f} mm，"
              f"自映射 {(twin == np.arange(len(verts))).sum()} 点")
    print(f"  骨 24 根，{len(LANDMARKS)} 类标志点（共 "
          f"{sum(2 if m['side'] else 1 for m in LANDMARKS)} 个点）")


if __name__ == "__main__":
    main()
