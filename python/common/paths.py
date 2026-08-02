"""项目内的路径约定。除本模块外没人算仓库根。

    inputs/     手工放进来的源数据（软链接也行）
    outputs/    全部产物，一律可删可重建
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT / "inputs"
OUTPUTS = ROOT / "outputs"

# 蹦床 FBX 的默认位置：先看 inputs/，再退回仓库上一级（原始放置处）。
TRAMP_FBX_NAMES = ("BengChuang_01.fbx",)
# dive 的 HSMR skin 拓扑。与 frames.bin 的 skin 顶点数必须一致。
DIVE_SKIN_OBJ = Path(
    "/Users/penghaotian/Documents/GIT/rtsp-h264-stitcher/weights/diving/"
    "hsmr_official/simplified_advanced_skin_1400.obj")


def display(path):
    """相对仓库根的短路径，打印用。"""
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def find_tramp_fbx(explicit=None):
    """蹦床源 FBX。`inputs/` 优先，其次仓库上一级目录。"""
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise SystemExit(f"FBX 不存在：{path}")
        return path.resolve()
    for name in TRAMP_FBX_NAMES:
        for folder in (INPUTS, ROOT.parent):
            candidate = folder / name
            if candidate.is_file():
                return candidate.resolve()
    raise SystemExit(f"找不到源 FBX，把它放进 {display(INPUTS)}/ 或用 --src 指定")


def resolve_metrics(frames_path, explicit=None):
    """`*_replay_frames.bin[.br]` 旁边的 `*_replay_metrics.json`。"""
    if explicit:
        path = Path(explicit).expanduser().resolve()
    else:
        stem = Path(frames_path).name.split("_replay_frames")[0]
        path = Path(frames_path).parent / f"{stem}_replay_metrics.json"
    if not path.is_file():
        raise SystemExit(f"找不到 metrics：{path}")
    return path


def clip_id(metrics_path):
    return Path(metrics_path).name.replace("_replay_metrics.json", "")


def dive_dir(clip):
    return OUTPUTS / f"dive_{clip}"


def latest_dive_dir(explicit=None):
    """最近一次 dive 提取的输出目录。"""
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_dir():
            raise SystemExit(f"目录不存在：{path}")
        return path
    found = sorted(OUTPUTS.glob("dive_*"))
    if not found:
        raise SystemExit(f"{display(OUTPUTS)} 下没有 dive_*，先跑 dive extract")
    return found[-1]


def find_video(metrics_path, clip, explicit=None):
    """叠加视频的底图。默认原始机位（下划线那版没有仪表盘）。"""
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"视频不存在：{path}")
        return path
    folder = Path(metrics_path).parent
    for name in (f"{clip.replace('-', '_')}.mp4", f"{clip}.mp4",
                 f"{clip}_dive.mp4"):
        candidate = folder / name
        if candidate.is_file():
            return candidate
    raise SystemExit(f"找不到底图视频，在 {folder} 下试过 <clip>.mp4 / <clip>_dive.mp4")


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
