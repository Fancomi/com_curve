# 国标体表标志点标注器

在 HSMR/SKEL 人体上点选 GB/T 17245-2004 表 1 的 20 个体表骨性标志，存成**原生顶点索引**。

```bash
python3 scripts/export_gb_landmark_asset.py     # 生成 outputs/gb_landmark_*.{json,bin}
python3 -m http.server 8731                     # 仓库根起服务（file:// 下 fetch 会被拦）
open http://127.0.0.1:8731/tools/gb_landmark_label/
```

标完点「💾 导出 JSON」，把 `gb_landmarks.json` 放到 `inputs/` 下。

## 不抽稀：索引就是 SKEL/SMPL 原生索引

页面渲染的是**稠密**网格：`skin_template_v` 6890 顶点、`skel_template_v` 247252 顶点。
标出来的 `vertex` 直接索引 SKEL 前向输出，**中间没有任何需要维护的对照表**。

早先那一版渲染 HSMR 的两个简化 OBJ（1379 / 3271），代价是抽稀没有成对保留顶点：

```
1379 子集里镜像还在子集内的：587 个（42.6%）
骨盆区 156 个顶点里只有 80 个有镜像
「x 取负找最近」的偏差：无镜像的 792 个点中位 18.9 mm、最大 48.7 mm
```

标左右成对的骨突时，对侧压根没有对应顶点 —— 这是放弃抽稀的直接原因。

## 镜像表 twin

导出脚本一并算出 `twin`（每个顶点的镜像顶点），页面用它显示对侧位置、做镜像与中线吸附。
不再用「x 取负找最近」那种近似：

| 层 | 来源 | 偏差 |
| --- | --- | --- |
| skin | pkl 自带的 `vert_sym_idxs`（合规对合，134 个中线点自映射） | 中位 4.69 mm，90% 8.88 mm |
| skel | 按骨构造 | 中位 4.30 mm，90% 10.23 mm，1239 点自映射 |

skel 没有自带表，构造方式是：**成对骨**的左右顶点一一对应（T-pose 下逐位镜像偏差中位
2.4–18 mm），所以用骨内下标偏移；`hand_r` 恰好是 `hand_l` 的 2 倍所以取模。**中线骨**
（pelvis / lumbar_body / thorax / head）在该骨**内部**做几何镜像 —— 限制在骨内很关键，
全网格找最近会跳到别的部位（中位 15.5 mm、最大 176 mm，而按骨做是 4.3 / 43 mm）。

SMPL 模板本身只是近似对称（`vert_sym_idxs` 配对的几何偏差中位 4.7 mm、中线点 x 最大
6.1 mm），这是模板的性质，不是这里能修的。

## 二进制格式

稠密 skel 走 JSON 数字文本要 28 MB，二进制只要 5.7 MB。布局记在 manifest 的
`offsets` 里，顺序 `positions f32 → faces u16/u32 → bone u8 → twin i32`，**每块起始
偏移补齐到 4 字节** —— JS 的 `new Int32Array(buffer, offset, n)` 要求 offset 是 4 的
倍数，而 `bone` 是 u8（skin 6890 个）会让 twin 落在奇数偏移上直接抛 RangeError。

法向不导出，页面 `computeVertexNormals()` 现算（省 3 MB）。

## 界面

单视口 + **自由轨迹球**。左键拖旋转、右键拖平移、滚轮缩放、`Q`/`E` 绕视线转、
`R` 复位、`F` 聚焦当前骨、`C` 剖切、`I` 隔离、`W` 线框，顶栏六个机位按钮只是起手式，
落位后照样自由转。

不用 `OrbitControls`：它把世界 `+Y` 锁成「上」，转到顶部会卡住、也翻不过去看脚底。
这里让 `camera.up` 跟旋转一起走（同一个四元数同时作用在 offset 和 up 上），
于是没有极点奇异 —— 「底」那个机位就是为了标 `足底` 用的。

显示开关分四组：**皮肤面／骨骼面**、**皮肤顶点／骨骼顶点**（两层各一个点云，
与面共用同一份 position buffer，所以点云里第 i 个点就是 `verts[i]`）、
**线框／SKEL 关节**、**剖切／中线面**。切换拾取层时会自动打开对应的顶点云，
免得「点得到但看不见是哪个点」。

拾取默认从**骨骼层**（24.7 万顶点）取 —— 国标要的是骨性标志。
左键既拖也点，靠位移阈值区分（≤4 px 算点击）。

## 剖切 ≠ 透明度

两件完全不同的事，之前那版叫「剖切」但实现的是背面剔除，确实只看得出明暗变化：

| | 做了什么 | 对拾取的影响 |
| --- | --- | --- |
| 透明度滑条 | 只改 `material.opacity`，前后两层几何**都还在** | 前层照样贡献 `frontDepth`，把后面的顶点挡在拾取之外 |
| ✂ 剖切 | GPU `clippingPlanes` 沿视线把靠镜头那半**移除** | 切掉的顶点不渲染、也不参与拾取（`distanceToPoint < 0` 直接跳过） |

裁剪面法向跟着相机走、过 orbit 点，滑条 `剖切深度` 调切面前后位置。实测正面机位下
`depth = 0` 时 247252 个顶点里 102575 个（41.5%）被移除 —— 是真的把几何分成两组，
不是画淡。

剖切开启时**不再用 `frontDepth` 过滤**：`raycaster` 不认 `clippingPlanes`，它命中的
会是已经被切掉的那层皮，`frontDepth` 偏浅会把切面后的骨突全挡住。

## 隔离相关骨会真正锁住拾取

开「🔍 隔离相关骨」（或按 `I`）时做两件事：

1. 骨骼几何只留相关骨的面，顶点云同步（单侧骨在 `bone_verts_indices` 里是连续区间，
   直接用 `drawRange`；`pelvis` 这种不分侧的退回建索引）
2. **`isolateSet` 限制拾取** —— 不在集合里的顶点直接跳过。「看不见」不等于「点不到」，
   这是关键的一条

**皮肤不动。** 骨突相对体表的位置是标注的重要参照，挡视线就调透明度或按 `C` 剖切。

各标志点隔离后的可拾取顶点数：内踝 99、桡骨茎突 236、桡骨 280、胫骨 845、
肩峰 989、足 2916、骨盆 4495、手 20598–41196、头 49599、胸廓 109176 —— 都非空。

## 拾取为什么不会穿过去

三道闸：

1. **先打 Mesh 射线拿最前面那层表面的深度**，比它深 5 cm（`PICK_MARGIN`）以上的顶点
   直接丢。人体闭合，正对镜头时前后两层顶点在屏幕上几乎重合。剖切开启时跳过这一闸
   （见上）。
2. **顶点法向过滤**：`normal · dir > 0.15` 的丢掉。阈值不是 0，因为轮廓线上的骨突
   （内踝、桡骨茎突）法向几乎垂直于视线。
3. **隔离模式的骨号过滤**与**剖切面过滤**（见上）。

评分用屏幕像素距离而不是三维垂距 —— 透视下三维垂距偏爱近处顶点，而用户点的是屏幕上
那个位置。深度项只作同像素时的次序。

注意 `Vector3.applyMatrix4` **内部已经做过透视除法**，结果就是 NDC、没有 `.w` 分量；
早先按 `clip.w` 手工再除一次拿到 NaN，`NaN < bestScore` 恒为 false，于是整个页面点不动。

## 迁移旧标注

稀疏口径的 JSON 用这个脚本迁到稠密索引（精确查表，不是最近点近似）：

```bash
python3 scripts/migrate_gb_landmarks.py ~/Downloads/gb_landmarks.json
```

子集顶点 `i` 就是稠密网格的 `mapping[i]`，两个 mapping 由 HSMR 权重目录提供。
实测 20 个点的坐标偏差最大 0.0065 mm（float32 往返的舍入）。页面载入时会拒绝
`skin_count` 不匹配的文件 —— 静默载入会把点放到别的部位。

## 已知的坑

- **T-pose 是跑 SKEL 前向拿的**（`poses=0, betas=0, trans=0`），不是随手取一帧、
  也不是 HSMR 的简化 OBJ —— 那两个 OBJ 是屈髋屈膝的姿态。零姿态下手臂水平外展
  （x 跨度 1.837 m ≈ y 跨度 1.791 m）、左右对称。
- 走前向还顺带解决了骨骼：`skel_template_v` 是**骨局部坐标**（各骨绕自己原点，
  pelvis 的 bbox 只有 0.16 m），不乘 24 个骨变换拼不成人；前向输出是摆好的，
  24 个关节也是白给的。
- **坐标轴是量出来的**：上 `+Y`、前 `+Z`、右 `-X`。依据是 toes 的 z 大于 pelvis、
  femur_r 的 x 为负。

## 数据来源

```
SKEL pkl     egl_render_haotian/src/hsmr_lite/skel/skel_male.pkl
             跑 RenderSKEL 前向拿 T-pose；bone_verts_indices / skin_weights 给骨归属；
             vert_sym_idxs 给 skin 的镜像表
hsmr_lite    egl_render_haotian/src（--hsmr-lite 指定，需要 torch + scipy）
```

`skel_female.pkl` 用不上 —— 它的 `skel_template_v` 只有 115010 个顶点，
dive 的 `skel_vertex_mapping` 最大索引 247217 直接越界，所以这套数据是 male 模型。

迁移脚本额外要 HSMR 权重目录的两个 `*_vertex_mapping.npy`。

## vendor 文件

`three.module.js` 不进库（`.gitignore` 里排除了），重新拿：

```bash
cp ../../../knowledge_work/label_mocap/smpl_web_viewer/public/vendor/three.module.js \
   tools/gb_landmark_label/
```

只需要 three 本体 —— 相机控制是自己写的，不依赖 `OrbitControls`。
