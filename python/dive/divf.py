"""DIVF replay 的二进制格式。除本模块外没人碰 struct 与字节偏移。

文件头 32 字节：magic `DIVF` + `<7I` = version, frame_count, skin_n, skel_n,
kp3d_n, fps, flags。之后每帧一条定长记录：

    逐帧头 64 字节（旧版 56，按总长度推断）
        +0   frame_idx  u32
        +4   status     u32     位 0 = has_mesh，位 1 = has_pose
        +8   cam_t      3f      crop 相机平移，z 是单目深度估计
        +20  focal      f
        +24  bbox       4f
        +40  patch      i32
    float32 顶点块：skin_n×3, skel_n×3, kp3d_n×3

**skel 与 kp3d 两块只参与步长计算，从不读取**：writer 目前把 kp3d 全写 0（没把
`PoseResult.kp3d` 搬进来），skel 也不是这条线要的东西。所以「根骨骼」用 rest pose 上
离骨盆最近的一撮顶点的均值代理，见 `align.pelvis_ids`。
"""
import struct
import subprocess
from pathlib import Path

import numpy as np

HEADER_BYTES = 32
FRAME_HEADER_BYTES = 64
LEGACY_FRAME_HEADER_BYTES = 56


def load_obj(path):
    """(顶点, 三角面)。顶点只用来选骨盆代理，几何以 frames.bin 为准。"""
    vertices, faces = [], []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("v "):
            parts = line.split()
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif line.startswith("f "):
            corners = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
            for k in range(1, len(corners) - 1):
                faces.append((corners[0], corners[k], corners[k + 1]))
    return np.asarray(vertices, np.float64), np.asarray(faces, np.int64)


def read(path):
    """读出未压缩的 DIVF 字节。`.br` 优先 brotli 模块，退回同名文件，最后退回 CLI。"""
    path = Path(path)
    raw = path.read_bytes()
    if raw[:4] == b"DIVF":
        return raw
    try:
        import brotli

        raw = brotli.decompress(raw)
    except Exception:
        name = path.name[:-3] if path.name.endswith(".br") else path.name
        for candidate in (path.parent / name, Path("/tmp") / name):
            if candidate.exists() and candidate.read_bytes()[:4] == b"DIVF":
                return candidate.read_bytes()
        out = Path("/tmp") / name
        run = subprocess.run(["brotli", "-d", "-f", "-o", str(out), str(path)],
                             capture_output=True, text=True)
        if run.returncode or not out.exists():
            raise SystemExit(f"无法解压 {path}（装 brotli 模块或 CLI）")
        raw = out.read_bytes()
    if raw[:4] != b"DIVF":
        raise SystemExit(f"不是 DIVF：{raw[:4]!r}")
    return raw


def header(raw):
    """解析文件头，附带 per（每帧字节数）与 pfh（逐帧头长度）。"""
    version, count, skin_n, skel_n, kp_n, fps, flags = struct.unpack_from(
        "<7I", raw, 4)
    body = 4 * 3 * (skin_n + skel_n + kp_n)
    per, pfh = FRAME_HEADER_BYTES + body, FRAME_HEADER_BYTES
    if len(raw) != HEADER_BYTES + count * per:
        legacy = LEGACY_FRAME_HEADER_BYTES + body
        if len(raw) != HEADER_BYTES + count * legacy:
            raise SystemExit(f"frames.bin 尺寸不符：{len(raw)}")
        per, pfh = legacy, LEGACY_FRAME_HEADER_BYTES
    return dict(version=version, n=count, skin_n=skin_n, skel_n=skel_n,
                kp_n=kp_n, fps=fps, flags=flags, per=per, pfh=pfh)


def record(raw, head, index):
    """一帧：crop 相机参数 + skin 顶点 + 是否可用。"""
    off = HEADER_BYTES + index * head["per"]
    status, = struct.unpack_from("<I", raw, off + 4)
    cam = np.array(struct.unpack_from("<3f", raw, off + 8), np.float64)
    focal, = struct.unpack_from("<f", raw, off + 20)
    bbox = np.array(struct.unpack_from("<4f", raw, off + 24), np.float64)
    patch, = struct.unpack_from("<i", raw, off + 40)
    skin = np.frombuffer(raw, "<f4", head["skin_n"] * 3,
                         off + head["pfh"]).reshape(head["skin_n"], 3).astype(
                             np.float64)
    usable = (bool(status & 1) and np.isfinite(skin).all()
              and not np.allclose(skin, 0))
    return dict(status=int(status), usable=usable, cam=cam, focal=float(focal),
                bbox=bbox, patch=int(patch), skin=skin)
