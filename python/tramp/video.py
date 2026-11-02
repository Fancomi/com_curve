"""蹦床线的四格对照视频。

两种用法共用一套 compose：
  同一场景比两种口径   `--pair mesh gb`
  两个文件比同一口径   `--reference before.fbx --compare mesh`

after 在右、before 在左（时间顺序＝阅读顺序）。两格都用虚线 + 小空心环叠出根骨骼位置：
before 格里虚线才是被约束成抛物线的那条，after 格里实线才是。不这么画读者看不出
"哪条被改、哪条被约束"。
"""
import cv2
import numpy as np

from ..common import grid as Grid
from ..common import paths as P
from ..common import raster as R
from . import curves as Cv
from . import scene as F
from . import stage as St


class Take:
    """一个已打开的场景 + 它的采样结果。两个文件对比时各持一个。"""

    def __init__(self, path, args, wanted, window=None):
        self.path = path
        self.manager, self.scene = F.load(path)
        self.sampler = Cv.Sampler(self.scene, wanted, args.body, args.bed,
                                  args.root_bone)
        self.stage = St.Stage(self.sampler)
        begin, end = window or (self.sampler.first, self.sampler.last)
        self.frames = np.arange(max(begin, self.sampler.first),
                                min(end, self.sampler.last) + 1)
        self.data = self.sampler.run(self.frames, wanted,
                                     extras=("volume", "contact", "bbox",
                                             "segments", "bed_low", "pelvis"))
        self.bed_rest = float(np.median(self.data["bed_low"]))
        self.stamps = [self.sampler.stamp(f) for f in self.frames]

    @property
    def count(self):
        return len(self.frames)

    def corners(self):
        """包围盒的两个角点，喂给 `frame_ranges`。"""
        return np.vstack([self.data["low"], self.data["high"]])

    def close(self):
        self.manager.Destroy()


def _panels(args, takes, pair, labels):
    """面板列表 + 横幅里那个差值的名字。"""
    if args.reference is None:
        take = takes[0]
        specs = [Grid.panel(St.MODE_CN[key], St.MODE_SUB[key], St.MARKER[key],
                            take.data[key], value=St.metres,
                            segments=take.data["segments"].get(key),
                            weights=_weights(take, key),
                            full_scale=St.MASS_FULL_SCALE)
                 for key in pair]
        gap = f"{St.MODE_CN[pair[1]]} - {St.MODE_CN[pair[0]]} 高度差"
        return specs, gap
    compare = args.compare or pair[0]
    specs = []
    # before 在左（source 1 是 --reference 给的那个文件），after 在右。
    for source, label, colour in ((1, labels[0], St.BEFORE_TINT),
                                  (0, labels[1], St.MARKER[compare])):
        take = takes[source]
        specs.append(Grid.panel(
            label, St.MODE_SUB[compare], colour, take.data[compare],
            source=source, aux=take.data["pelvis"], aux_colour=St.AUX_TINT,
            aux_label="骨盆", value=St.metres,
            segments=take.data["segments"].get(compare),
            weights=_weights(take, compare), full_scale=St.MASS_FULL_SCALE))
    return specs, f"{labels[1]} - {labels[0]} 高度差"


def _weights(take, key):
    """体段质心圆的权重：归一化的质量比例，各格共用满刻度才能比大小。"""
    if key not in Cv.SEGMENTED:
        return None
    masses = take.sampler.bodies.gb_mass
    return masses / masses.sum()


def run(args):
    pair = tuple(args.pair)
    if args.reference is None and pair[0] == pair[1]:
        raise SystemExit("--pair 的两个口径要不同")
    compare = args.compare or pair[0]
    wanted = pair if args.reference is None else (compare,)

    out = P.OUTPUTS / "tramp"
    tag = (f"{pair[0]}_{pair[1]}" if args.reference is None
           else f"before_after_{compare}")
    name = args.name or f"render_{tag}.mp4"
    frames_dir = out / f"frames_{tag}"
    if frames_dir.is_dir():
        for stale in frames_dir.glob("*.png"):
            stale.unlink()
    frames_dir.mkdir(parents=True, exist_ok=True)

    takes = []
    try:
        src = P.find_tramp_fbx(args.src)
        main = Take(src, args, wanted, args.frames)
        takes.append(main)
        labels = args.labels or ["before", "after"]
        if args.reference is not None:
            reference = Take(args.reference.expanduser().resolve(), args, wanted,
                             (int(main.frames[0]), int(main.frames[-1])))
            takes.append(reference)
            if reference.count != main.count:
                raise SystemExit(f"两个文件的帧数不同（{main.count} vs "
                                 f"{reference.count}），没法逐帧对比")
            print(f"{src.name}: 渲染 f{main.frames[0]}..{main.frames[-1]}"
                  f"（{main.count} 帧），对比 {args.reference.name}（{compare}）")
        else:
            print(f"{src.name}: 渲染 f{main.frames[0]}..{main.frames[-1]}"
                  f"（{main.count} 帧），口径 {pair[0]} vs {pair[1]}")
        if main.sampler.bodies is not None:
            print(f"  体段刚体：{main.sampler.bodies.voxel_count} 体素")

        specs, gap_label = _panels(args, takes, pair, labels)
        # 相机范围要覆盖**所有** take：两个版本共用一套刻度，否则"修正后高了 75 cm"
        # 会被各自的自动取景吃掉。
        cameras = St.cameras([t.corners() for t in takes],
                             [s["track"] for s in specs], args.height)
        total = sum(c.width for c in cameras.values()) * len(specs)
        print(f"  {'，'.join(f'{St.VIEW_CN[k]} {c.width}px' for k, c in cameras.items())}"
              f"；成片 {total}×{args.height + 38}")

        painter = R.TextPainter()
        if not painter.available:
            print("  （没找到中文字体，标签退回 ASCII）")
        # 世界单位是厘米，刻度每米一条、标签写"1m"，所以 label_step 是 1/100 步长。
        scale = {"unit": "m", "step": Cv.CM_PER_M, "label_step": 1.0,
                 "top": 100, "datum": main.bed_rest, "tick": 26}
        views = St.views()
        sources = sorted({spec["source"] for spec in specs})

        for index in range(main.count):
            bases = {(source, key): takes[source].stage.base(
                camera, takes[source].stamps[index])
                for source in sources for key, camera in cameras.items()}
            phase = "触床" if main.data["contact"][index] > 1e-6 else "腾空"
            delta = (specs[1]["track"][index, 1] - specs[0]["track"][index, 1])
            extra = (f"目标 g = -{args.gravity / Cv.CM_PER_M:.2f} m/s/s"
                     if args.reference is not None
                     else f"体积 {main.data['volume'][index] / 1000.0:.1f} L")
            text = (f"f{int(main.frames[index]):>4d} / {int(main.frames[-1])}"
                    f"   {index / main.sampler.fps:5.2f} s   {phase}"
                    f"   {gap_label} {delta:+6.2f} cm   {extra}")
            frame = Grid.compose(bases, specs, views, cameras, index, painter,
                                 text, scale, trail_span=St.TRAIL_SPAN)
            cv2.imwrite(str(frames_dir / f"{index + 1:04d}.png"), frame)
            if (index + 1) % 50 == 0 or index + 1 == main.count:
                print(f"    {index + 1}/{main.count}")

        video = R.encode(frames_dir, out / name, main.sampler.fps,
                         keep=args.keep_frames)
        print(f"  wrote {P.display(video)} "
              f"({video.stat().st_size / 1e6:.1f} MB)")
    finally:
        for take in reversed(takes):
            take.close()
