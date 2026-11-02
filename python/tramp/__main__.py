"""蹦床线的统一入口：`python -m python.tramp STEP [选项…]`

步骤（可逗号分隔按序执行）：

    extract    FBX      -> outputs/tramp/curves.{csv,json,npz} + markers.fbx
    plot       CSV      -> outputs/tramp/{curves,trajectory}.png
    render     FBX      -> outputs/tramp/render_<A>_<B>.mp4
    retarget   FBX      -> outputs/tramp/retarget/<target>.{fbx,csv,json,npz,png}
    spin       retarget/mesh.fbx -> outputs/tramp/spin/mesh.{fbx,json,npz}（整体旋转修复）
    spinvideo  spin 产物 -> outputs/tramp/spin/mesh.mp4（修复前后，跟踪质心三视角）
    verify     校验全部产物

`extract` 是其余步骤的前提（它写的 JSON 里有触床区间）；`spin` 要先有 `retarget`。
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from ..common import paths as P
from ..common import plotstyle as PS
from ..common import signal as Sig
from ..common import table as T
from . import curves as Cv
from . import markers, plots, retarget, scene as F, spin, stage as St

STEPS = ("extract", "plot", "render", "retarget", "spin", "spinvideo", "verify")
OUT = P.OUTPUTS / "tramp"
CURVES_CSV = OUT / "curves.csv"
CURVES_JSON = OUT / "curves.json"
RETARGET_DIR = OUT / "retarget"
SPIN_DIR = OUT / "spin"


def _load_scene(args):
    return F.load(P.find_tramp_fbx(args.src))


def _columns(data, keys, fps):
    """CSV 列：位置、速度、两两高度差、体积、触床。"""
    dt = 1.0 / fps
    columns = [("frame", data["frames"]),
               ("time_s", (data["frames"] - data["frames"][0]) * dt)]
    for key in keys:
        columns += T.vector_columns(key, data[key], "cm")
    for key in keys:
        speed = Sig.derivative(data[key] / Cv.CM_PER_M, dt)
        columns += [(f"{key}_vy_mps", speed[:, 1]),
                    (f"{key}_speed_mps", np.linalg.norm(speed, axis=1))]
    for index, left in enumerate(keys):
        for right in keys[index + 1:]:
            columns.append((f"{right}_minus_{left}_y_cm",
                            data[right][:, 1] - data[left][:, 1]))
    columns += [("volume_l", data["volume"] / 1000.0),
                ("contact_weight", data["contact"]),
                ("body_low_y_cm", data["foot"]),
                ("bed_low_y_cm", data["bed_low"])]
    return columns


def step_extract(args):
    """采两条曲线，写 CSV/JSON/NPZ + 一个带动画的 FBX。"""
    manager, scene = _load_scene(args)
    try:
        keys = tuple(args.modes)
        sampler = Cv.Sampler(scene, keys, args.body, args.bed, args.root_bone)
        print(f"{P.display(P.find_tramp_fbx(args.src))}: "
              f"{sampler.last - sampler.first + 1} 帧 @ {sampler.fps:g} fps，"
              f"口径 {' '.join(keys)}")
        facts = sampler.facts()
        print(f"  人体 {facts['body']}：{facts['vertices']} 顶点 / "
              f"{facts['triangles']} 三角形，闭合={facts['closed_manifold']}，"
              f"{facts['bones_weighted']} 根骨有权重")
        if "rigid_bodies" in facts:
            rigid = facts["rigid_bodies"]
            print(f"  体段刚体：{rigid['voxel_count']} 体素 @ "
                  f"{rigid['voxel_pitch_cm']} cm，{rigid['voxel_volume_l']:.3f} L vs "
                  f"整网 {rigid['mesh_volume_l']:.3f} L "
                  f"（{rigid['voxel_vs_mesh_pct']:+.2f}%），"
                  f"全身 {rigid['total_mass_kg']:.1f} kg")

        data = sampler.run(wanted=keys,
                           extras=("volume", "contact", "foot", "bed_low"))
        contacts = sampler.contacts(data)
        gravity = Cv.gravity_report(data, keys, contacts, sampler.fps, args.trim)
        stats = Cv.pairwise_stats(data, keys)
        print(f"  {len(contacts)} 次触床，{len(gravity['flights'])} 段腾空")
        for key in keys:
            block = gravity["per_curve"][key]
            print(f"    {key:7s} g = {block['gravity_mean']:+.2f} ± "
                  f"{block['gravity_std']:.2f} m/s²   残差 "
                  f"{block['rms_mean']:.1f} cm")
        for name, entry in stats.items():
            print(f"    {name:18s} ΔY 均值 {entry['dy_mean_cm']:+7.2f}  σ "
                  f"{entry['dy_std_cm']:5.2f}  最大 {entry['dy_max_abs_cm']:6.2f} cm")

        print(f"  wrote {P.display(T.write_table(CURVES_CSV, _columns(data, keys, sampler.fps)))}")
        meta = {
            "source": str(P.find_tramp_fbx(args.src)),
            "unit": "cm", "up_axis": "Y",
            "fps": sampler.fps,
            "frames": [int(sampler.first), int(sampler.last)],
            "modes": {key: Cv.MODE_DOC[key] for key in keys},
            "root_bone": args.root_bone,
            "scene": facts,
            "body_volume_l": {
                "min": round(float(data["volume"].min() / 1000), 3),
                "max": round(float(data["volume"].max() / 1000), 3),
                "mean": round(float(data["volume"].mean() / 1000), 3)},
            "bed_rest_y_cm": round(float(np.median(data["bed_low"])), 3),
            "contacts": contacts,
            "gravity": gravity,
            "pairwise": stats,
        }
        print(f"  wrote {P.display(P.write_json(CURVES_JSON, meta))}")
        np.savez(OUT / "curves.npz", **{k: v for k, v in data.items()
                                        if k != "segments"})
        path = markers.write(OUT / "markers.fbx", data["frames"],
                            {key: data[key] for key in keys}, sampler.mode)
        print(f"  wrote {P.display(path)}")
        if args.with_scene:
            merged = markers.write(OUT / "markers_in_scene.fbx", data["frames"],
                                   {key: data[key] for key in keys}, sampler.mode,
                                   scene=scene, manager=manager)
            print(f"  wrote {P.display(merged)}")
    finally:
        manager.Destroy()


def step_plot(args):
    """CSV + JSON -> 两张对照图。"""
    data = T.read_table(CURVES_CSV)
    meta = P.read_json(CURVES_JSON)
    cjk = PS.setup()
    print(f"  wrote {P.display(plots.curves_figure(data, meta, OUT / 'curves.png', cjk))}")
    print(f"  wrote {P.display(plots.trajectory_figure(data, meta, OUT / 'trajectory.png', cjk))}")
    for target in sorted(RETARGET_DIR.glob("*.json")):
        stem = target.stem
        table = T.read_table(RETARGET_DIR / f"{stem}.csv")
        info = P.read_json(target)
        path = plots.retarget_figure(table, info, meta,
                                    RETARGET_DIR / f"{stem}.png", cjk)
        print(f"  wrote {P.display(path)}")


def step_render(args):
    """四格对照视频。`--modes A B` 选两种口径，`--reference` 比两个文件。"""
    from . import video

    video.run(args)


def step_retarget(args):
    """反算根骨骼。`--targets` 选盯哪条曲线。"""
    if not CURVES_JSON.is_file():
        raise SystemExit(f"先跑 extract：需要 {P.display(CURVES_JSON)} 里的触床区间")
    contacts = P.read_json(CURVES_JSON)["contacts"]
    src = P.find_tramp_fbx(args.src)
    RETARGET_DIR.mkdir(parents=True, exist_ok=True)

    for target in args.targets:
        manager, scene = F.load(src)
        try:
            sampler = Cv.Sampler(scene, (target,), args.body, args.bed,
                                 args.root_bone, need_bed=False)
            layer = F.base_layer(scene)
            print(f"{src.name}: {sampler.last - sampler.first + 1} 帧 @ "
                  f"{sampler.fps:g} fps，目标 {target}")
            retarget.check_root_chain(sampler.root_bone, sampler.mode,
                                      np.arange(sampler.first, sampler.last + 1))

            before = sampler.run(wanted=(target,), extras=("pelvis",))
            frames = before["frames"]
            delta, report = retarget.solve(before[target], frames, contacts,
                                           sampler.fps, args.gravity)
            magnitude = np.linalg.norm(delta, axis=1)
            touching = np.zeros(len(frames), bool)
            for start, stop in contacts:
                touching[(frames >= start) & (frames <= stop)] = True
            print(f"  {len(report)} 段腾空需要修正，触床帧 Δ 恒为 "
                  f"{np.abs(delta[touching]).max():.1e} cm")
            print(f"  |Δroot| 均值 {magnitude.mean():.2f} cm，最大 "
                  f"{magnitude.max():.2f} cm（f{int(frames[magnitude.argmax()])}）")
            for entry in report:
                print(f"    f{entry['frames'][0]:4d}..{entry['frames'][1]:4d} "
                      f"位移 ≤{entry['shift_max_cm']:6.1f} cm  apex "
                      f"{entry['apex_before_m']:.2f} -> {entry['apex_after_m']:.2f} m"
                      f"  离床速度 {entry['takeoff_before_mps']:+.2f} -> "
                      f"{entry['takeoff_after_mps']:+.2f} m/s")

            retarget.apply(sampler.root_bone, layer, frames, delta, sampler.mode)
            scene.GetAnimationEvaluator().Reset()

            # 重采一遍确认被盯曲线真的落在抛物线上。这是校验而不是装饰：闭式解的
            # 正确性依赖"平移根 = 平移该曲线"，重采是对它的端到端确认。
            after = sampler.run(wanted=(target,), extras=("pelvis",))
            error = float(np.abs(after[target] - (before[target] + delta)).max())
            print(f"  重采校验：{target} 与目标最大差 {error:.2e} cm")
            if error > 1e-3:
                raise SystemExit(f"修正后 {target} 没落在目标上（差 {error:.3e} cm）")

            dt = 1.0 / sampler.fps
            fits = [Sig.fit_parabola(
                after[target][int(np.searchsorted(frames, a)):
                              int(np.searchsorted(frames, b)) + 1, 1] / Cv.CM_PER_M,
                dt) for a, b in (e["frames"] for e in report)]
            print(f"  修正后腾空段拟合 {np.mean([f[0] for f in fits]):+.4f} ± "
                  f"{np.std([f[0] for f in fits]):.4f} m/s²，残差 "
                  f"{np.mean([f[1] for f in fits]) * Cv.CM_PER_M:.1e} cm")

            fbx_path = F.save(manager, scene, RETARGET_DIR / f"{target}.fbx")
            print(f"  wrote {P.display(fbx_path)} "
                  f"({fbx_path.stat().st_size / 1e6:.1f} MB)")

            speed = Sig.derivative(before[target] / Cv.CM_PER_M, dt)
            fixed_speed = Sig.derivative(after[target] / Cv.CM_PER_M, dt)
            columns = ([("frame", frames),
                        ("time_s", (frames - frames[0]) * dt)]
                       + T.vector_columns(target, before[target], "cm")
                       + T.vector_columns("root", before["pelvis"], "cm")
                       + T.vector_columns(f"fixed_{target}", after[target], "cm")
                       + T.vector_columns("fixed_root", after["pelvis"], "cm")
                       + T.vector_columns("shift", delta, "cm")
                       + [("shift_cm", magnitude),
                          ("vy_mps", speed[:, 1]),
                          ("fixed_vy_mps", fixed_speed[:, 1])])
            print(f"  wrote {P.display(T.write_table(RETARGET_DIR / f'{target}.csv', columns))}")
            meta = {
                "source": str(src), "target": target, "unit": "cm",
                "fps": sampler.fps,
                "frames": [int(sampler.first), int(sampler.last)],
                "gravity_cm_s2": args.gravity,
                "root_bone": args.root_bone,
                "method": "closed form: translating the root rigidly translates "
                          "the whole body, so delta_root = target - actual; the "
                          "target is a parabola pinned at both ends of a flight",
                "contact_delta_max_cm": float(np.abs(delta[touching]).max()),
                "shift_mean_cm": round(float(magnitude.mean()), 3),
                "shift_max_cm": round(float(magnitude.max()), 3),
                "resample_error_cm": error,
                "gravity_after_mps2": round(float(np.mean([f[0] for f in fits])), 5),
                "gravity_after_rms_cm": float(
                    np.mean([f[1] for f in fits]) * Cv.CM_PER_M),
                "flights": report,
            }
            print(f"  wrote {P.display(P.write_json(RETARGET_DIR / f'{target}.json', meta))}")
            np.savez(RETARGET_DIR / f"{target}.npz", frames=frames,
                     before=before[target], root=before["pelvis"], delta=delta,
                     after=after[target], fixed_root=after["pelvis"])
        finally:
            manager.Destroy()


def step_spin(args):
    """整体旋转修复：每段腾空的全身角动量守恒。输入是 `retarget` 的产物。"""
    if not CURVES_JSON.is_file():
        raise SystemExit(f"先跑 extract：需要 {P.display(CURVES_JSON)} 里的腾空区间")
    windows = P.read_json(CURVES_JSON)["gravity"]["flights"]
    src = RETARGET_DIR / f"{args.spin_target}.fbx"
    if not src.is_file():
        raise SystemExit(f"先跑 retarget --targets {args.spin_target}：需要 {P.display(src)}")
    SPIN_DIR.mkdir(parents=True, exist_ok=True)
    manager, scene = F.load(src)
    try:
        sampler = Cv.Sampler(scene, ("gb", args.spin_target), args.body, args.bed,
                             args.root_bone, need_bed=False)
        layer = F.base_layer(scene)
        frames = np.arange(sampler.first, sampler.last + 1)
        dt = 1.0 / sampler.fps
        root = sampler.root_bone
        retarget.check_root_chain(root, sampler.mode, frames)
        pre, xyz = F.euler_pre_rotation(root)
        if not xyz:
            raise SystemExit(f"{root.GetName()} 的旋转顺序不是 XYZ，写回键的公式不适用")
        print(f"{src.name}: {len(frames)} 帧 @ {sampler.fps:g} fps，"
              f"{len(windows)} 段腾空，枢轴 = {args.spin_target} 质心")

        stamps = [sampler.stamp(f) for f in frames]
        body = spin.body_of(sampler.bodies)
        rotation, translation = spin.sample(sampler.bodies, stamps)
        centres = body.centres(rotation, translation)
        before = sampler.run(wanted=(args.spin_target,), extras=("volume",))
        pivot = before[args.spin_target] / Cv.CM_PER_M
        theta, report = spin.solve(body, rotation, centres, pivot, frames, windows, dt)
        for entry in report:
            print(f"    f{entry['frames'][0]:4d}..{entry['frames'][1]:4d} 修正 ≤"
                  f"{entry['correction_max_deg']:6.1f}°  L 偏差 "
                  f"{entry['L_before_spread']:6.2f} -> {entry['L_after_spread']:5.2f}")

        euler = retarget.read_keys(root, "LclRotation", layer, frames, sampler.mode)
        moved = retarget.read_keys(root, "LclTranslation", layer, frames, sampler.mode)
        euler, moved = spin.root_keys(theta, euler, moved, pre,
                                      before[args.spin_target])
        retarget.edit_keys(root, "LclRotation", layer, frames, sampler.mode,
                           lambda i, c, _old: euler[i, c])
        retarget.edit_keys(root, "LclTranslation", layer, frames, sampler.mode,
                           lambda i, c, _old: moved[i, c])
        scene.GetAnimationEvaluator().Reset()

        # 重采一遍：质心不动、旋转等于 θ，是对"整体转 = 只改根骨骼"的端到端确认。
        after = sampler.run(wanted=(args.spin_target,), extras=("volume",))
        com_gap = float(np.abs(after[args.spin_target] - before[args.spin_target]).max())
        fixed_rotation, fixed_translation = spin.sample(sampler.bodies, stamps)
        want = np.einsum("nij,nsjk->nsik",
                         spin.Rot.from_rotvec(theta).as_matrix(), rotation)
        rotation_gap = float(np.abs(fixed_rotation - want).max())
        print(f"  重采校验：质心最大偏移 {com_gap:.1e} cm，段旋转与 θ 最大差 "
              f"{rotation_gap:.1e}")
        # FBX 曲线键是 float32：Euler 值累计到上千度，单精度只剩 1e-4° 量级，旋转矩阵
        # 元素差 ~4e-6 是存储精度而不是写错。写错（Pre 乘反、顺序错）会差到 1e-1 以上。
        if com_gap > 1e-3 or rotation_gap > 1e-4:
            raise SystemExit("写回的根骨骼键没有还原出求解的整体旋转")

        fbx_path = F.save(manager, scene, SPIN_DIR / f"{args.spin_target}.fbx")
        print(f"  wrote {P.display(fbx_path)} ({fbx_path.stat().st_size / 1e6:.1f} MB)")
        lateral = spin.lateral_axis(sampler.bodies)
        L0, tilt0, rate0 = spin.traces(body, rotation, centres, lateral, dt)
        L1, tilt1, rate1 = spin.traces(
            body, fixed_rotation, body.centres(fixed_rotation, fixed_translation),
            lateral, dt)
        meta = {
            "source": str(src), "pivot": args.spin_target, "unit": "kg m^2/s",
            "fps": sampler.fps, "frames": [int(frames[0]), int(frames[-1])],
            "method": "per flight, rotate the whole body about the pivot by theta(t) "
                      "so that the whole-body angular momentum is one constant "
                      "vector; theta = 0 at both ends; only Root_M keys change",
            "lambdas": list(spin.LAMBDAS),
            "segment_mass_kg": round(body.total, 3),
            "com_shift_cm": com_gap,
            "correction_max_deg": max(e["correction_max_deg"] for e in report),
            "flights": report,
        }
        print(f"  wrote {P.display(P.write_json(SPIN_DIR / f'{args.spin_target}.json', meta))}")
        np.savez(SPIN_DIR / f"{args.spin_target}.npz", frames=frames, theta=theta,
                 fps=sampler.fps,
                 com=before[args.spin_target], windows=np.array(windows),
                 L_before=L0, L_after=L1, tilt_before=tilt0, tilt_after=tilt1,
                 rate_before=rate0, rate_after=rate1)
    finally:
        manager.Destroy()


def step_spinvideo(args):
    """spin 的修复前后对比视频。"""
    from . import spinvideo

    spinvideo.run(args, RETARGET_DIR / f"{args.spin_target}.fbx",
                  SPIN_DIR / f"{args.spin_target}.fbx",
                  SPIN_DIR / f"{args.spin_target}.npz",
                  SPIN_DIR / f"{args.spin_target}.mp4")


def step_verify(args):
    from . import verify

    verify.run(args)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m python.tramp", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("steps", help="逗号分隔：" + ",".join(STEPS))
    parser.add_argument("--src", type=Path, default=None,
                        help="源 FBX（默认 inputs/BengChuang_01.fbx 或仓库上一级）")
    parser.add_argument("--body", default=None, help="人体网格节点名")
    parser.add_argument("--bed", default=None, help="蹦床床面节点名")
    parser.add_argument("--root-bone", default="Root_M")
    parser.add_argument("--modes", nargs="+", default=list(Cv.MODES),
                        choices=Cv.MODES, help="extract 要采哪几条曲线")
    parser.add_argument("--trim", type=int, default=6,
                        help="重力拟合时每段腾空两端各切掉几帧")
    parser.add_argument("--with-scene", action="store_true",
                        help="extract 额外导一份并进原场景的 FBX（13 MB）")
    # retarget
    parser.add_argument("--targets", nargs="+", default=["mesh"],
                        choices=Cv.MODES, help="retarget 盯哪几条曲线")
    parser.add_argument("--gravity", type=float, default=retarget.GRAVITY_CM,
                        help="重力大小 cm/s²")
    # spin
    parser.add_argument("--spin-target", default="mesh", choices=Cv.MODES,
                        help="spin 修哪个 retarget 产物，也是整体旋转的枢轴")
    # render
    parser.add_argument("--pair", nargs=2, default=["mesh", "gb"],
                        choices=Cv.MODES, metavar=("A", "B"),
                        help="render 对比哪两种口径")
    parser.add_argument("--reference", type=Path, default=None,
                        help="render 改成「视角 × 两个文件」，用它当 before")
    parser.add_argument("--compare", default=None, choices=Cv.MODES,
                        help="--reference 模式下两边各画哪条曲线（默认跟 --pair[0]）")
    parser.add_argument("--labels", nargs=2, default=None,
                        metavar=("BEFORE", "AFTER"))
    parser.add_argument("--name", default=None, help="render 输出文件名")
    parser.add_argument("--height", type=int, default=620, help="单格像素高度")
    parser.add_argument("--frames", type=int, nargs=2, default=None,
                        metavar=("FIRST", "LAST"), help="只处理这一段（调试用）")
    parser.add_argument("--keep-frames", action="store_true",
                        help="保留 PNG 序列")
    # verify
    parser.add_argument("--check-frames", type=int, nargs="*",
                        default=[100, 200, 300], help="蒙特卡洛对照的帧")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    actions = {"extract": step_extract, "plot": step_plot, "render": step_render,
               "retarget": step_retarget, "spin": step_spin,
               "spinvideo": step_spinvideo, "verify": step_verify}
    chosen = [s.strip() for s in args.steps.split(",") if s.strip()]
    unknown = [s for s in chosen if s not in actions]
    if unknown:
        raise SystemExit(f"未知步骤 {unknown}，可选：{', '.join(STEPS)}")
    for step in chosen:
        actions[step](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
