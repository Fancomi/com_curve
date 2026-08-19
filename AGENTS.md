# com_curve —— 给 agent 的路由表

**先定位你要改的那条线，只读那一列的文件。** 全库扫一遍是浪费——两条线互不交叉。

| 你的任务提到… | 那条线 | 入口 | 要读的文件 |
| --- | --- | --- | --- |
| 蹦床、`BengChuang_01.fbx`、FBX、骨骼、蒙皮、root/mesh/mujoco/anthro、反算根骨骼、抛物线 | tramp | `scripts/tramp.sh` | `python/tramp/*` |
| 跳水、`bin.br`、DIVF、replay、HSMR、单目、尺度标定、叠加源视频 | dive | `scripts/dive.sh` | `python/dive/*` |
| 体积重心公式、抛物线拟合、CSV、光栅器、四格布局、配色 | 共用底座 | — | `python/common/*` |
| 国标、GB/T 17245、体段质量比、质心相对位置、表 6/表 7 | 两条线共用 | — | `python/common/gb17245.py` |
| 体表骨性标志、髂前上棘/胫骨点/内踝点、标注顶点索引、SKEL 稠密网格 | 标注工具 | 见该目录 README | `tools/gb_landmark_label/*` |

改共用底座前**先确认两条线都还对**：`./scripts/tramp.sh verify` 是唯一的自动化闸门。

## 两条线各自的入口点

```
tramp   extract  -> curves.{csv,json,npz} + markers.fbx    python/tramp/__main__.py:step_extract
        retarget -> retarget/<target>.{fbx,csv,json,npz}   python/tramp/__main__.py:step_retarget
        plot     -> curves.png trajectory.png              python/tramp/plots.py
        render   -> render_*.mp4                           python/tramp/video.py
        verify                                             python/tramp/verify.py

dive    extract  -> curve.{csv,json,npz}                   python/dive/__main__.py:step_extract
                （默认：fit_flight 后贝塞尔轻度重采样曲线）
        plot     -> height/trajectory/scale.png            python/dive/plots.py
        render   -> render.mp4（正交四格）                  python/dive/__main__.py:step_render
        overlay  -> overlay.mp4（叠源视频）                 python/dive/__main__.py:step_overlay
        批量     -> outputs/batch_<目录名>.log              scripts/dive_batch.sh
```

## 国标 gb 口径：两条线不是同一个量

同名但端点来源不同，**别把两条的数字混着比**：

```
tramp gb   端点是绑定骨架的**关节中心**（Hip_L/Knee_L…）。国标表 1 要体表骨突，
           所以有系统偏差：大腿偏短 3.3 个百分点身高、小腿偏长 2.6。
           适配层在 python/tramp/segments.py 的 GB_SEGMENT_MAP（14 段）。
dive  gb   端点是**手工标注的真骨突**，与表 1 严格一致，15 段。
           标注在 inputs/gb_landmarks.json，解析在 python/dive/landmarks.py。
```

表值只有一份：`python/common/gb17245.py`。要改表值改那里，两条线一起变。
`tools/gb_landmark_label/` 是标注页面，产出的 JSON 存**顶点索引**（SKEL 原生索引，
无抽稀、无中间映射），所以逐帧 `verts[index]` 就是那个解剖点。

## 加一种重心口径要改哪里

tramp：只改 `python/tramp/curves.py`——`MODES` 加一项、`Sampler._point` 加一支、
`MODE_DOC` 写一句定义。CSV 列、JSON 口径、图表配色、视频面板、CLI 选项全从 `MODES` 派生。

dive：`python/dive/align.py` 的 `MODES` 加一项 + `Clip._place` 里多算一条。注意 dive
的口径可能**缺**（`gb` 要标注文件），所以下游一律遍历 `clip.modes` 而不是 `A.MODES`，
JSON 里 `modes` 只列实际有的那几条，`plots.curve_keys(meta)` 从 JSON 反推。

**不要**在别处再写一遍分派逻辑——那正是这次整理消掉的东西（原先五个脚本各写了一遍）。

## 边界（越过它就是设计错误）

- **只有 `python/tramp/scene.py` 顶层 `import fbx`。** 其余模块拿到的一律是 numpy 数组，
  所以没有 SDK 的机器也能跑 dive 那条线、能读 CSV 画图。少数需要 SDK 常量的函数
  （`bodies.bind_pose`、`markers.write`、`verify.bind_pose_error`）在函数体内 import。
- **`python/common/` 不认识业务口径。** 它没有"重心""腾空""触床"的概念，也不 import
  `tramp` / `dive`。要往里加东西，先问它对两条线是否都成立。
- **`python/common/paths.py` 是唯一算仓库根的地方。** 别在别处写 `Path(__file__).parents[n]`。
- **单位不隐式换算。** tramp 是厘米、dive 是米，`common` 里所有函数都单位无关，由调用方
  传 `unit` / `label_step` 之类的参数对齐。混用不会报错，只会让数字差 100 倍。

## 改动后必须跑的

```bash
./scripts/tramp.sh extract,plot,verify        # 约 40 s，45 项断言
./scripts/dive.sh extract --frames <某个 bin.br> --check   # 重投影必须是 0.000000 px
```

全量（约 4 分 + 45 s）：`./scripts/tramp_full.sh`、`./scripts/dive_full.sh <frames>`。
整目录：`./scripts/dive_batch.sh DIR --skip <pat>`。

## 数据

源数据不进库。tramp 找 `inputs/BengChuang_01.fbx`（软链接也行），其次仓库上一级。
dive 要这些：`*_replay_frames.bin.br`、同名 `*_replay_metrics.json`（可缺，缺了退 hsmr
尺度）、HSMR 权重目录（`python/common/paths.py:HSMR_DIR`，里面有 skin 拓扑 OBJ 与两个
vertex mapping）。`gb` 口径另需 `inputs/gb_landmarks.json` 与 SKEL pkl（跑前向拿稠密
T-pose，`--hsmr-lite` 指定；缺任何一个就自动跳过这条曲线，不报错）。

`outputs/` 全部可删可重建，不要往里手工放东西。**例外**：`inputs/gb_landmarks.json`
是手工标注的产物，重标一次要几十分钟，别删。

## 口径与结论写在哪

**别在代码注释里重复结论**，也别新建总结文档。数值口径、为什么这么选、试过什么方案而
放弃，全在 `README.md`。代码注释只写"这一行为什么不能简化"。

已经踩过的坑（改到相关处先读 README 对应段落）：正面朝向要从骨架量、光源必须每相机一个、
体素射线交点要去重、`ear_canal` 不能逐帧现算、速度要用 `masked_derivative`、腾空窗口要
从 metrics 事件收窄到含最高点的一条弧、dive 的 `s` 是全片冻结的米/像素（物理上≈Z/fx，
但不是逐帧深度，也不是 FOV/体型）、Hiragino Sans GB 缺 U+2212 与 U+00B2。
