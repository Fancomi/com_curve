// 国标体表标志点标注器 —— 在稠密 SKEL T-pose 上点选顶点索引。
//
// **顶点索引就是 SKEL/SMPL 原生索引**（skin 6890 / skel 247252），不抽稀、不建中间
// 映射。下游拿 SKEL 前向输出后 `verts[index]` 直接可用。早先那版渲染 HSMR 的简化
// OBJ（1379/3271），抽稀没有成对保留顶点——1379 里只有 42.6% 的点镜像还在子集内，
// 骨盆区 156 个点里只有 80 个，标左右成对的骨突时对侧压根没有对应顶点。
//
// 相机是自由轨迹球，不用 OrbitControls：后者把世界 +Y 锁成「上」，转到顶部会卡住、
// 也没法翻过去看脚底。这里让 `camera.up` 跟着旋转一起走（同一个四元数作用在 offset
// 和 up 上），于是没有极点奇异。
import * as THREE from 'three';

const MANIFEST = '../../outputs/gb_landmark_asset.json';
const OUT_NAME = 'gb_landmarks.json';
const ROT_SPEED = 0.006;            // 弧度 / 像素
const DRAG_SLOP = 4;                // 小于这个位移算点击而不是拖动（像素）
const PICK_MARGIN = 0.05;           // 拾取容差：允许比最前面那层表面深 5 cm
const MIDLINE_TOL = 0.004;          // 中线判据：|x| < 4 mm
const HOME = { eye: [1.5, 0.2, 2.3], up: [0, 1, 0], target: [0, -0.32, 0] };

const $ = (id) => document.getElementById(id);
const status = (msg) => { $('status').textContent = msg; };

const COLORS = ['#ff5252', '#ffab40', '#ffee58', '#9ccc65', '#26c6da',
                '#42a5f5', '#7e57c2', '#ec407a', '#8d6e63', '#78909c',
                '#00e676', '#ff6e40'];

const state = {
  asset: null,
  layers: {},         // skin / skel -> { positions, faces, bone, twin, count }
  targets: [],
  active: 0,
  picks: {},          // id -> { vertex, from }
  pickFrom: 'skel',   // 国标骨性标志绝大多数在骨骼上，默认从骨骼拾取
  isolate: false,
  isolateSet: null,   // 隔离时允许拾取的骨号；null = 不限制
};

// ---- 场景 ----------------------------------------------------------------

const canvas = $('c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setClearColor(0x05070a, 1);
// 剖切用 GPU 裁剪面（真的把几何切开，不是改透明度），要显式打开。
renderer.localClippingEnabled = true;

const scene = new THREE.Scene();
scene.add(new THREE.HemisphereLight(0xddeeff, 0x223344, 0.75));
scene.add(new THREE.AmbientLight(0xffffff, 0.4));
// 头灯跟着相机：自由视角下固定方向光会让转到背面时整个人是黑的。
const headLight = new THREE.DirectionalLight(0xffffff, 0.55);
scene.add(headLight);

const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 100);
const orbit = new THREE.Vector3(...HOME.target);

function resetCamera() {
  camera.position.set(...HOME.eye);
  camera.up.set(...HOME.up);
  orbit.set(...HOME.target);
  camera.lookAt(orbit);
}
resetCamera();

// 自由轨迹球：把「相机绕 target 的偏移」和「相机的 up」用同一个四元数转。
// 水平拖 = 绕当前 up 转；垂直拖 = 绕当前右手轴转。没有极点，能一路翻到脚底。
function rotateCamera(dx, dy) {
  const offset = camera.position.clone().sub(orbit);
  const right = new THREE.Vector3().crossVectors(camera.up, offset).normalize();
  const yaw = new THREE.Quaternion().setFromAxisAngle(
    camera.up.clone().normalize(), -dx * ROT_SPEED);
  const pitch = new THREE.Quaternion().setFromAxisAngle(right, -dy * ROT_SPEED);
  const spin = yaw.multiply(pitch);
  offset.applyQuaternion(spin);
  camera.up.applyQuaternion(spin).normalize();
  camera.position.copy(orbit).add(offset);
  camera.lookAt(orbit);
}

function panCamera(dx, dy) {
  const offset = camera.position.clone().sub(orbit);
  const scale = offset.length() * 0.0016;
  const right = new THREE.Vector3().crossVectors(camera.up, offset)
    .normalize().multiplyScalar(-dx * scale);
  const up = camera.up.clone().normalize().multiplyScalar(dy * scale);
  orbit.add(right).add(up);
  camera.position.add(right).add(up);
  camera.lookAt(orbit);
}

function zoomCamera(delta) {
  const offset = camera.position.clone().sub(orbit);
  const factor = Math.exp(delta * 0.0013);
  const length = THREE.MathUtils.clamp(offset.length() * factor, 0.15, 12);
  camera.position.copy(orbit).add(offset.setLength(length));
}

function rollCamera(sign) {
  const axis = orbit.clone().sub(camera.position).normalize();
  camera.up.applyQuaternion(
    new THREE.Quaternion().setFromAxisAngle(axis, sign * 0.08)).normalize();
  camera.lookAt(orbit);
}

let skinMesh = null, skelMesh = null, jointPoints = null;
const points = { skin: null, skel: null };     // 两层各一个顶点云
const markGroup = new THREE.Group();
scene.add(markGroup);

// 剖切面：法向朝相机，常数随相机位置更新，于是「切掉靠近镜头的那半」。
// 这跟透明度完全不是一件事 —— 透明度是把前表面画淡，几何还在，前面那层的顶点照样
// 挡着拾取；剖切是把它从渲染和 raycaster 里**移除**，切面后面的骨突直接暴露出来。
const clipPlane = new THREE.Plane(new THREE.Vector3(0, 0, -1), 0);
const clipState = { on: false, depth: 0 };

function updateClip() {
  if (!clipState.on) return;
  const normal = orbit.clone().sub(camera.position).normalize();
  clipPlane.normal.copy(normal);
  // 过 orbit 点、再沿视线推 depth：滑条负值多切一点，正值少切一点。
  clipPlane.constant = -normal.dot(orbit) - clipState.depth;
}

function applyClip() {
  const planes = clipState.on ? [clipPlane] : [];
  for (const mesh of [skinMesh, skelMesh, points.skin, points.skel]) {
    if (!mesh) continue;
    mesh.material.clippingPlanes = planes;
    mesh.material.needsUpdate = true;
  }
  updateClip();
}

const cursor = new THREE.Mesh(
  new THREE.SphereGeometry(1, 16, 12),
  new THREE.MeshBasicMaterial({ color: 0x2fd6e0, wireframe: true }));
cursor.visible = false;
scene.add(cursor);

// 镜像提示：把光标顶点的 twin（真镜像顶点，不是几何最近点）画成蓝色空心球 + 连线。
// 中线点看两球是否重合就知道偏没偏；成对点看对侧落在哪。
const mirrorHint = new THREE.Group();
const mirrorBall = new THREE.Mesh(
  new THREE.SphereGeometry(1, 16, 12),
  new THREE.MeshBasicMaterial({ color: 0x5b8dff, wireframe: true }));
const mirrorLine = new THREE.Line(
  new THREE.BufferGeometry().setFromPoints(
    [new THREE.Vector3(), new THREE.Vector3()]),
  new THREE.LineBasicMaterial({ color: 0x5b8dff, transparent: true,
                                opacity: 0.6 }));
mirrorHint.add(mirrorBall, mirrorLine);
mirrorHint.visible = false;
scene.add(mirrorHint);

const midPlane = new THREE.Mesh(
  new THREE.PlaneGeometry(1.2, 2.2),
  new THREE.MeshBasicMaterial({ color: 0x3b6ea8, transparent: true,
                                opacity: 0.13, side: THREE.DoubleSide,
                                depthWrite: false }));
midPlane.rotation.y = Math.PI / 2;
midPlane.position.set(0, -0.32, 0.02);
midPlane.visible = false;
scene.add(midPlane);

function layerOf(from) { return state.layers[from]; }

function vertexAt(from, index) {
  const layer = layerOf(from);
  return new THREE.Vector3(layer.positions[index * 3],
                           layer.positions[index * 3 + 1],
                           layer.positions[index * 3 + 2]);
}

function vertexOf(pick) { return vertexAt(pick.from, pick.vertex); }

// twin 表来自导出脚本：skin 用 pkl 自带的 `vert_sym_idxs`，skel 按骨构造
// （成对骨用骨内下标偏移，中线骨在骨内做几何镜像）。中线上的点 twin 指向自己。
function twinOf(from, index) {
  const twin = layerOf(from).twin[index];
  return twin >= 0 ? twin : index;
}

function showMirror(from, index, size) {
  const here = vertexAt(from, index);
  const twin = twinOf(from, index);
  const there = vertexAt(from, twin);
  mirrorBall.position.copy(there);
  mirrorBall.scale.setScalar(size * 1.4);
  mirrorLine.geometry.setFromPoints([here, there]);
  const gap = here.distanceTo(there);
  mirrorHint.visible = twin !== index && gap > 1e-4;
  $('mirror-info').textContent = twin === index
    ? `v${index} 在中线上（twin 指向自己）`
    : `x = ${(here.x * 1000).toFixed(1)} mm　twin v${twin}　`
      + `间距 ${(gap * 1000).toFixed(1)} mm`;
}

function resize() {
  const stage = $('stage');
  const w = stage.clientWidth, h = stage.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
addEventListener('resize', resize);

function render() {
  requestAnimationFrame(render);
  headLight.position.copy(camera.position);
  headLight.target.position.copy(orbit);
  headLight.target.updateMatrixWorld();
  updateClip();
  renderer.render(scene, camera);
}

// ---- 资源 ----------------------------------------------------------------

// 二进制布局由导出脚本的 manifest 给出：positions f32 -> faces u16/u32 ->
// bone u8 -> twin i32，全部紧邻。稠密 skel 有 24.7 万顶点，走 JSON 数字文本要
// 28 MB，二进制只要 5.7 MB。
async function loadLayer(block, base) {
  const raw = await (await fetch(new URL(block.file, base))).arrayBuffer();
  const off = block.offsets;
  const IndexArray = block.index_type === 'u16' ? Uint16Array : Uint32Array;
  return {
    count: block.count,
    positions: new Float32Array(raw, off.positions, block.count * 3),
    faces: new IndexArray(raw, off.faces, block.faces * 3),
    bone: new Uint8Array(raw, off.bone, block.count),
    twin: new Int32Array(raw, off.twin, block.count),
  };
}

function buildMesh(layer, colour, opacity) {
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(layer.positions, 3));
  geom.setIndex(new THREE.BufferAttribute(layer.faces, 1));
  geom.computeVertexNormals();
  const material = new THREE.MeshStandardMaterial({
    color: colour, roughness: 0.65, metalness: 0.05,
    transparent: true, opacity, side: THREE.DoubleSide,
    depthWrite: opacity > 0.9 });
  return new THREE.Mesh(geom, material);
}

// 顶点云与面共用同一个 position buffer（同一份数据、同一套索引），所以点云里第 i 个
// 点就是 verts[i]，不存在两套编号。
function buildPoints(layer, colour, size) {
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(layer.positions, 3));
  const cloud = new THREE.Points(geom, new THREE.PointsMaterial({
    color: colour, size, sizeAttenuation: false }));
  cloud.visible = false;
  return cloud;
}

async function load() {
  status('加载 manifest …');
  const base = new URL(MANIFEST, location.href);
  const response = await fetch(base);
  if (!response.ok) {
    status(`加载失败 ${response.status} —— 先跑 `
      + 'scripts/export_gb_landmark_asset.py');
    return;
  }
  const asset = await response.json();
  state.asset = asset;

  status(`加载网格（skel ${(asset.skel.bytes / 1e6).toFixed(1)} MB）…`);
  state.layers.skin = await loadLayer(asset.skin, base);
  state.layers.skel = await loadLayer(asset.skel, base);

  skinMesh = buildMesh(state.layers.skin, 0xc8d4e8, 0.30);
  skelMesh = buildMesh(state.layers.skel, 0xe8dcc8, 1.0);
  scene.add(skinMesh, skelMesh);

  points.skin = buildPoints(state.layers.skin, 0x9fd0ff, 2);
  points.skel = buildPoints(state.layers.skel, 0xffd08a, 2);
  scene.add(points.skin, points.skel);

  const jointGeom = new THREE.BufferGeometry();
  jointGeom.setAttribute('position',
    new THREE.Float32BufferAttribute(asset.joints.flat(), 3));
  jointPoints = new THREE.Points(jointGeom, new THREE.PointsMaterial({
    color: 0xff3366, size: 9, sizeAttenuation: false }));
  jointPoints.visible = false;
  scene.add(jointPoints);

  buildTargets();
  resize();
  selectTarget(0);
  status(`就绪 —— 稠密 T-pose，skin ${asset.skin.count} / `
    + `skel ${asset.skel.count} 顶点（原生索引，无抽稀）`);
}

// 把 12 类展开成 20 个具体目标。
function buildTargets() {
  state.targets = [];
  state.asset.landmarks.forEach((mark, index) => {
    const colour = COLORS[index % COLORS.length];
    if (mark.side) {
      for (const side of ['R', 'L']) {
        state.targets.push({ ...mark, colour, side,
          id: `${mark.key}_${side}`,
          label: `${mark.cn}·${side === 'R' ? '右' : '左'}` });
      }
    } else {
      state.targets.push({ ...mark, colour, side: null, id: mark.key,
        label: mark.cn });
    }
  });
}

// ---- 列表与详情 ----------------------------------------------------------

function renderList() {
  const list = $('lm-list');
  list.innerHTML = '';
  state.targets.forEach((target, index) => {
    const pick = state.picks[target.id];
    const row = document.createElement('div');
    row.className = 'lm' + (index === state.active ? ' on' : '')
      + (pick ? ' done' : '');
    const dot = document.createElement('span');
    dot.className = 'dot';
    dot.style.background = target.side === 'L' ? 'transparent' : target.colour;
    dot.style.borderColor = target.colour;
    const name = document.createElement('span');
    name.className = 'nm';
    name.textContent = target.label;
    const st = document.createElement('span');
    st.className = 'st';
    st.textContent = pick
      ? `${pick.from === 'skel' ? 'K' : 'S'}${pick.vertex}` : '未标';
    row.append(dot, name, st);
    row.onclick = () => selectTarget(index);
    list.appendChild(row);
  });
  $('progress').textContent =
    `${Object.keys(state.picks).length} / ${state.targets.length}`;
}

function selectTarget(index) {
  state.active = ((index % state.targets.length) + state.targets.length)
    % state.targets.length;
  const target = state.targets[state.active];
  const counts = boneVertexCounts(target);
  $('detail').innerHTML =
    `<span class="k">${target.label}</span>　`
    + `<span class="b">${boneLabelOf(target)}</span>`
    + `<br>${target.hint}`
    + `<br><span class="dim">该骨顶点：骨骼 ${counts.skel} / 皮肤 ${counts.skin}`
    + '</span>';
  const pick = state.picks[target.id];
  $('picked').innerHTML = !pick
    ? '尚未标注'
    : `已标 <code>${pick.from === 'skel' ? '骨骼' : '皮肤'} v${pick.vertex}</code>`
      + `　<code>${vertexOf(pick).toArray()
        .map((v) => v.toFixed(4)).join(', ')}</code>`;
  if (!pick) {
    $('mirror-info').textContent = '';
  } else if (!target.side) {
    const here = vertexOf(pick);
    const twin = twinOf(pick.from, pick.vertex);
    $('mirror-info').textContent = twin === pick.vertex
      ? `x = ${(here.x * 1000).toFixed(1)} mm　twin 指向自己 = 在中线上 ✓`
      : `x = ${(here.x * 1000).toFixed(1)} mm　twin v${twin} 不是自己 `
        + '→ 不在中线上，可按「⊥ 吸附到中线」';
  } else {
    const other = state.picks[`${target.key}_${target.side === 'R' ? 'L' : 'R'}`];
    const twin = twinOf(pick.from, pick.vertex);
    $('mirror-info').textContent = other
      ? (other.vertex === twin
        ? `对侧 v${other.vertex} 正是本点的 twin ✓ 完全对称`
        : `对侧 v${other.vertex}，本点 twin 是 v${twin} —— 两侧不是镜像对`)
      : `对侧未标　本点 twin = v${twin}，可按「⇄ 镜像到对侧」`;
  }
  applyIsolate();
  renderList();
}

function boneVertexCounts(target) {
  const wanted = new Set(boneIndices(target.bone, target.side));
  const out = {};
  for (const name of ['skin', 'skel']) {
    let count = 0;
    for (const bone of state.layers[name].bone) if (wanted.has(bone)) count += 1;
    out[name] = count;
  }
  return out;
}

// ---- 隔离相关骨 ----------------------------------------------------------

// 骨名带 _r/_l 后缀，标志点的 bone 字段给的是不分侧的词干（scapula、tibia…），
// 所以按前缀匹配、再按当前目标的 side 过滤。
function boneIndices(stem, side) {
  const wanted = [];
  state.asset.bone_names.forEach((name, index) => {
    if (!name.startsWith(stem)) return;
    if (side && (name.endsWith('_r') || name.endsWith('_l'))) {
      if (name.endsWith(side === 'R' ? '_r' : '_l')) wanted.push(index);
    } else {
      wanted.push(index);
    }
  });
  return wanted;
}

function boneLabelOf(target) {
  return target.side
    ? `${target.bone}_${target.side.toLowerCase()}` : target.bone;
}

// 隔离时同时做两件事，否则「看不见但点得到」还是会选错：
//   1) 骨骼几何只留相关骨的面（顶点云也只留那些点）
//   2) `state.isolateSet` 限制拾取——不在集合里的顶点直接跳过（这是关键）
// **皮肤不动**：骨突相对体表的位置是标注的重要参照；挡视线就调透明度或剖切。
function applyIsolate() {
  if (!skelMesh) return;
  const target = state.targets[state.active];
  const layer = state.layers.skel;
  if (!state.isolate) {
    skelMesh.geometry.setIndex(new THREE.BufferAttribute(layer.faces, 1));
    if (points.skel) {
      points.skel.geometry.setIndex(null);
      points.skel.geometry.setDrawRange(0, layer.count);
    }
    state.isolateSet = null;
    $('bone-name').textContent = '';
    return;
  }
  const wanted = new Set(boneIndices(target.bone, target.side));
  state.isolateSet = wanted;
  const bone = layer.bone, faces = layer.faces;
  const kept = [];
  for (let f = 0; f < faces.length; f += 3) {
    if (wanted.has(bone[faces[f]])) kept.push(faces[f], faces[f + 1], faces[f + 2]);
  }
  skelMesh.geometry.setIndex(kept);
  // 顶点云同样只画相关骨的点。单侧骨的顶点在 `bone_verts_indices` 里是连续区间
  // （实测 talus_r 是 6168..6266），所以用 drawRange 就够；不分侧的目标（pelvis）
  // 或将来跨多骨时退回建索引。
  if (points.skel) {
    const ranges = [...wanted].map((bone) => state.asset.bone_ranges[bone]);
    const contiguous = ranges.length === 1;
    const geometry = points.skel.geometry;
    if (contiguous) {
      geometry.setIndex(null);
      geometry.setDrawRange(ranges[0][0], ranges[0][1] - ranges[0][0]);
    } else {
      const picked = [];
      for (let i = 0; i < layer.count; i += 1) if (wanted.has(bone[i])) picked.push(i);
      geometry.setIndex(picked);
      geometry.setDrawRange(0, picked.length);
    }
  }
  $('bone-name').textContent = `隔离中：${boneLabelOf(target)}`
    + `　三角 ${kept.length / 3}　只能点中这根骨`;
}

// ---- 拾取 ----------------------------------------------------------------

const raycaster = new THREE.Raycaster();

// 拾取分三道闸，缺哪一道都会「穿过去」：
//
//   1) 先打一次 **Mesh 射线**拿到最前面那层表面的深度，比它深 PICK_MARGIN 以上的
//      顶点直接丢。人体是闭合的，正对镜头时前后两层顶点在屏幕上几乎重合。
//   2) 再用**顶点法向**滤掉背对镜头的顶点。轮廓线上的骨突（内踝、桡骨茎突）法向
//      几乎垂直视线，所以阈值放 0.15 而不是 0。
//   3) 隔离模式下按 `isolateSet` 过滤骨号 —— 「看不见」不等于「点不到」。
//
// 评分用**屏幕像素距离**：透视下三维垂距偏爱近处顶点，而用户点的是屏幕上那个位置。
// 深度项只作同像素时的次序。
const pickMatrix = new THREE.Matrix4();
const pickPoint = new THREE.Vector3();
const pickClip = new THREE.Vector3();
const pickDelta = new THREE.Vector3();
const pickNormal = new THREE.Vector3();

function pickVertex(ndc) {
  raycaster.setFromCamera(ndc, camera);
  const isSkin = state.pickFrom === 'skin';
  const layer = state.layers[state.pickFrom];
  const mesh = isSkin ? skinMesh : skelMesh;
  if (!mesh || !layer) return { index: -1, pixels: 0 };
  const array = layer.positions;
  const normalAttr = mesh.geometry.attributes.normal;
  const normals = normalAttr ? normalAttr.array : null;
  // 隔离只作用在骨骼层；皮肤层没有「相关骨」的概念那么强。
  const gate = (!isSkin && state.isolateSet) ? state.isolateSet : null;

  // 网格被隐藏时拿不到「最前面那层」，退回不做深度过滤。
  // 剖切开启时也不能用它：raycaster 不认 clippingPlanes，命中的会是已经被切掉的
  // 那层皮，frontDepth 偏浅，切面后面的骨突全被挡住。
  let frontDepth = Infinity;
  if (mesh.visible && !clipState.on) {
    const hits = raycaster.intersectObject(mesh, false);
    if (hits.length) frontDepth = hits[0].distance;
  }

  // matrixWorldInverse 由 renderer.render 更新；鼠标事件可能落在两帧之间，
  // 所以显式刷一次，否则刚转过相机时投影会用上一帧的矩阵。
  camera.updateMatrixWorld();
  pickMatrix.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
  const origin = raycaster.ray.origin, dir = raycaster.ray.direction;
  const element = renderer.domElement;
  const ratio = renderer.getPixelRatio();
  const halfW = element.width / ratio * 0.5;
  const halfH = element.height / ratio * 0.5;
  let best = -1, bestScore = Infinity, bestPixels = 0;

  for (let i = 0; i < layer.count; i += 1) {
    if (gate && !gate.has(layer.bone[i])) continue;
    pickPoint.set(array[i * 3], array[i * 3 + 1], array[i * 3 + 2]);
    // 剖切掉的那半也不该能点到——`distanceToPoint > 0` 是被保留的一侧。
    if (clipState.on && clipPlane.distanceToPoint(pickPoint) < 0) continue;
    pickDelta.copy(pickPoint).sub(origin);
    const along = pickDelta.dot(dir);
    if (along <= 0) continue;
    if (along > frontDepth + PICK_MARGIN) continue;
    if (normals) {
      pickNormal.set(normals[i * 3], normals[i * 3 + 1], normals[i * 3 + 2]);
      if (pickNormal.dot(dir) > 0.15) continue;
    }
    // `Vector3.applyMatrix4` 内部已做透视除法，结果就是 NDC（没有 .w）。
    pickClip.copy(pickPoint).applyMatrix4(pickMatrix);
    const dx = (pickClip.x - ndc.x) * halfW;
    const dy = (pickClip.y - ndc.y) * halfH;
    const pixels = Math.hypot(dx, dy);
    const score = pixels + along;
    if (score < bestScore) {
      bestScore = score; best = i; bestPixels = pixels;
    }
  }
  return { index: best, pixels: bestPixels };
}

function ndcAt(event) {
  const rect = $('stage').getBoundingClientRect();
  return new THREE.Vector2(
    ((event.clientX - rect.left) / rect.width) * 2 - 1,
    -((event.clientY - rect.top) / rect.height) * 2 + 1);
}

// ---- 鼠标：左键既拖也点，靠位移阈值区分 ----------------------------------

let drag = null;

canvas.addEventListener('pointerdown', (event) => {
  if (!state.asset) return;
  canvas.setPointerCapture(event.pointerId);
  drag = { button: event.button, x: event.clientX, y: event.clientY, moved: 0 };
});

canvas.addEventListener('pointermove', (event) => {
  if (!state.asset) return;
  if (drag) {
    const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
    drag.moved += Math.abs(dx) + Math.abs(dy);
    if (drag.button === 0) rotateCamera(dx, dy);
    else if (drag.button === 2 || drag.button === 1) panCamera(dx, dy);
    drag.x = event.clientX; drag.y = event.clientY;
    if (drag.moved > DRAG_SLOP) {
      cursor.visible = false; mirrorHint.visible = false;
    }
    return;
  }
  const hit = pickVertex(ndcAt(event));
  if (hit.index < 0) { cursor.visible = false; mirrorHint.visible = false; return; }
  const size = Number($('mark-size').value);
  cursor.position.copy(vertexAt(state.pickFrom, hit.index));
  cursor.scale.setScalar(size * 1.4);
  cursor.visible = true;
  showMirror(state.pickFrom, hit.index, size);
});

canvas.addEventListener('pointerup', (event) => {
  if (!drag) return;
  const wasClick = drag.button === 0 && drag.moved <= DRAG_SLOP;
  drag = null;
  if (!wasClick || !state.asset) return;
  const hit = pickVertex(ndcAt(event));
  if (hit.index < 0) {
    status(state.isolateSet ? '隔离模式下这里没有相关骨的顶点' : '这里没有顶点');
    return;
  }
  const target = state.targets[state.active];
  state.picks[target.id] = { vertex: hit.index, from: state.pickFrom };
  refreshMarks();
  selectTarget(state.active);
  const x = vertexAt(state.pickFrom, hit.index).x;
  const twin = twinOf(state.pickFrom, hit.index);
  status(`${target.label} ← ${state.pickFrom === 'skin' ? '皮肤' : '骨骼'}`
    + ` v${hit.index}（原生索引）　屏幕 ${hit.pixels.toFixed(1)} px`
    + (target.side ? `　twin v${twin}`
      : `　x = ${(x * 1000).toFixed(1)} mm`
        + (twin === hit.index ? '（在中线上 ✓）' : '（不在中线上）')));
});

canvas.addEventListener('wheel', (event) => {
  event.preventDefault();
  zoomCamera(event.deltaY);
}, { passive: false });

canvas.addEventListener('contextmenu', (event) => event.preventDefault());

// ---- 标记 ----------------------------------------------------------------

function refreshMarks() {
  markGroup.clear();
  const size = Number($('mark-size').value);
  for (const target of state.targets) {
    const pick = state.picks[target.id];
    if (!pick) continue;
    const here = vertexOf(pick);
    const mesh = new THREE.Mesh(
      new THREE.SphereGeometry(size, 18, 14),
      new THREE.MeshBasicMaterial({ color: target.colour }));
    mesh.position.copy(here);
    markGroup.add(mesh);
    // 中线点若 twin 不指向自己，把 twin 位置也画出来——一眼看出偏了多少。
    const twin = twinOf(pick.from, pick.vertex);
    if (!target.side && twin !== pick.vertex) {
      const ghost = new THREE.Mesh(
        new THREE.SphereGeometry(size, 14, 10),
        new THREE.MeshBasicMaterial({ color: target.colour, wireframe: true }));
      ghost.position.copy(vertexAt(pick.from, twin));
      markGroup.add(ghost);
    }
  }
}

// ---- 镜像与中线 ----------------------------------------------------------

// 直接用 twin 表，不再「x 取负找最近」——稠密网格里每个顶点都有确定的镜像顶点。
$('btn-mirror').onclick = () => {
  const target = state.targets[state.active];
  if (!target.side) { status('这个标志点不分左右'); return; }
  const pick = state.picks[target.id];
  if (!pick) { status('先标当前这一侧'); return; }
  const twin = twinOf(pick.from, pick.vertex);
  if (twin === pick.vertex) {
    status('这个点的 twin 指向自己（在中线上），没有对侧');
    return;
  }
  const twinId = `${target.key}_${target.side === 'R' ? 'L' : 'R'}`;
  state.picks[twinId] = { vertex: twin, from: pick.from };
  refreshMarks();
  renderList();
  const gap = vertexOf(pick).distanceTo(vertexAt(pick.from, twin));
  status(`${twinId} ← v${twin}（twin 表）　两点间距 `
    + `${(gap * 1000).toFixed(1)} mm`);
};

// 中线吸附：twin 指向自己的顶点才严格在中线上。在附近找一个这样的点。
$('btn-snap').onclick = () => {
  const target = state.targets[state.active];
  if (target.side) { status('这个标志点分左右，中线吸附不适用'); return; }
  const pick = state.picks[target.id];
  if (!pick) { status('先标一个点'); return; }
  const layer = layerOf(pick.from);
  const here = vertexOf(pick);
  let best = -1, bestDist = Infinity;
  const point = new THREE.Vector3();
  for (let k = 0; k < layer.count; k += 1) {
    if (layer.twin[k] !== k) continue;              // 只认 twin 指向自己的
    if (state.isolateSet && !state.isolateSet.has(layer.bone[k])) continue;
    point.set(layer.positions[k * 3], layer.positions[k * 3 + 1],
              layer.positions[k * 3 + 2]);
    const dist = point.distanceToSquared(here);
    if (dist < bestDist) { bestDist = dist; best = k; }
  }
  if (best < 0) { status('这一层没有 twin 自映射的中线顶点'); return; }
  state.picks[target.id] = { vertex: best, from: pick.from };
  refreshMarks();
  selectTarget(state.active);
  status(`${target.label} 吸附到 v${best}　`
    + `x = ${(layer.positions[best * 3] * 1000).toFixed(1)} mm，`
    + `移动 ${(Math.sqrt(bestDist) * 1000).toFixed(1)} mm`);
};

$('btn-clear').onclick = () => {
  const target = state.targets[state.active];
  delete state.picks[target.id];
  refreshMarks();
  selectTarget(state.active);
  status(`已清除 ${target.label}`);
};

// ---- 存取 ----------------------------------------------------------------

$('btn-save').onclick = () => {
  const marks = {};
  for (const target of state.targets) {
    const pick = state.picks[target.id];
    if (!pick) continue;
    const twin = twinOf(pick.from, pick.vertex);
    marks[target.id] = {
      vertex: pick.vertex,              // SKEL/SMPL 原生索引，无中间映射
      mesh: pick.from,                  // skin(6890) | skel(247252)
      twin,
      xyz: vertexOf(pick).toArray().map((v) => Number(v.toFixed(5))),
      cn: target.label, gb_key: target.key, side: target.side,
      bone: target.bone,
    };
  }
  const done = Object.keys(marks).length;
  const payload = {
    standard: 'GB/T 17245-2004 表 1 人体体段划分分界点（体表骨性标志）',
    note: 'vertex 是 SKEL/SMPL 原生索引（skin 6890 / skel 247252），'
      + '无抽稀、无中间映射；下游 verts[index] 直接可用',
    pose: state.asset.source.pose,
    axes: state.asset.axes,
    skin_count: state.layers.skin.count,
    skel_count: state.layers.skel.count,
    complete: done === state.targets.length,
    picked: done,
    total: state.targets.length,
    landmarks: marks,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)],
                        { type: 'application/json' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = OUT_NAME;
  link.click();
  URL.revokeObjectURL(link.href);
  status(`已导出 ${OUT_NAME}（${done}/${state.targets.length} 个点）`
    + '　→ 放到 inputs/ 下');
};

$('btn-load').onclick = () => $('json-input').click();
$('json-input').onchange = async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  const parsed = JSON.parse(await file.text());
  // 拒绝稀疏口径的旧文件：索引对不上，静默载入会把点放到别的部位。
  if (parsed.skin_count && parsed.skin_count !== state.layers.skin.count) {
    status(`这个 JSON 是 skin_count=${parsed.skin_count} 的旧稀疏口径，`
      + '先跑 scripts/migrate_gb_landmarks.py 迁到稠密索引');
    return;
  }
  state.picks = {};
  for (const [id, entry] of Object.entries(parsed.landmarks || {})) {
    state.picks[id] = { vertex: entry.vertex, from: entry.mesh || 'skin' };
  }
  refreshMarks();
  selectTarget(state.active);
  status(`已载入 ${Object.keys(state.picks).length} 个点`);
};

// ---- 开关 ----------------------------------------------------------------

function toggle(id, apply) {
  const button = $(id);
  button.onclick = () => {
    button.classList.toggle('on');
    apply(button.classList.contains('on'));
  };
}
toggle('t-skin', (on) => { if (skinMesh) skinMesh.visible = on; });
toggle('t-skel', (on) => { if (skelMesh) skelMesh.visible = on; });
toggle('t-skin-verts', (on) => { if (points.skin) points.skin.visible = on; });
toggle('t-skel-verts', (on) => { if (points.skel) points.skel.visible = on; });
toggle('t-joints', (on) => { if (jointPoints) jointPoints.visible = on; });
toggle('t-isolate', (on) => { state.isolate = on; applyIsolate(); });
toggle('t-midplane', (on) => { midPlane.visible = on; });

// 线框：看清三角划分与顶点密度（稠密 skel 有 12.7 万面，拉近看才有意义）。
toggle('t-wire', (on) => {
  for (const mesh of [skinMesh, skelMesh]) {
    if (!mesh) continue;
    mesh.material.wireframe = on;
    mesh.material.needsUpdate = true;
  }
});

// 剖切：GPU 裁剪面沿视线把靠近镜头的那半**移除**（不是画淡）。与透明度的区别：
// 透明度让前表面半透，几何还在、拾取仍被它的 frontDepth 挡着；剖切之后切面后方的
// 骨突直接暴露，也能点到（拾取里同样按裁剪面过滤了）。滑条调切面深浅。
toggle('t-clip', (on) => {
  clipState.on = on;
  applyClip();
  status(on ? '剖切开启 —— 切掉靠镜头那半，切面后的顶点可点；滑条调深浅'
    : '剖切关闭');
});
$('clip-depth').oninput = (event) => {
  clipState.depth = Number(event.target.value);
  updateClip();
};

$('skin-op').oninput = (event) => {
  if (!skinMesh) return;
  const value = Number(event.target.value);
  skinMesh.material.opacity = value;
  skinMesh.material.depthWrite = value > 0.9;
};
$('skel-op').oninput = (event) => {
  if (!skelMesh) return;
  const value = Number(event.target.value);
  skelMesh.material.opacity = value;
  skelMesh.material.depthWrite = value > 0.9;
};
$('vert-size').oninput = (event) => {
  const size = Number(event.target.value);
  for (const cloud of [points.skin, points.skel]) {
    if (cloud) cloud.material.size = size;
  }
};
$('mark-size').oninput = () => refreshMarks();

for (const [id, from] of [['p-skin', 'skin'], ['p-skel', 'skel']]) {
  $(id).onclick = () => {
    state.pickFrom = from;
    $('p-skin').classList.toggle('on', from === 'skin');
    $('p-skel').classList.toggle('on', from === 'skel');
    // 切换拾取层时把对应的顶点云打开，免得「点得到但看不见是哪个点」。
    const cloud = points[from];
    if (cloud && !cloud.visible) {
      $(from === 'skin' ? 't-skin-verts' : 't-skel-verts').click();
    }
    status(`拾取来源：${from === 'skin' ? '皮肤' : '骨骼'}顶点`
      + `（${state.layers[from].count} 个）`);
  };
}

// 六个正交机位只是**起手式**，落位后照样可以自由转——不锁方向。
const PRESETS = {
  front: { eye: [0, -0.32, 2.6], up: [0, 1, 0] },
  back: { eye: [0, -0.32, -2.6], up: [0, 1, 0] },
  left: { eye: [2.6, -0.32, 0], up: [0, 1, 0] },
  right: { eye: [-2.6, -0.32, 0], up: [0, 1, 0] },
  top: { eye: [0, 2.3, 0.001], up: [0, 0, -1] },
  bottom: { eye: [0, -2.9, 0.001], up: [0, 0, 1] },
};
for (const [name, spec] of Object.entries(PRESETS)) {
  const button = $(`v-${name}`);
  if (!button) continue;
  button.onclick = () => {
    orbit.set(...HOME.target);
    camera.position.set(...spec.eye);
    camera.up.set(...spec.up);
    camera.lookAt(orbit);
    status(`机位：${button.textContent.trim()}（之后可自由转）`);
  };
}

// 聚焦到当前标志点所在的骨：稠密网格下手工找骨突很费劲。
$('v-focus').onclick = () => {
  const target = state.targets[state.active];
  const wanted = new Set(boneIndices(target.bone, target.side));
  const layer = state.layers.skel;
  const centre = new THREE.Vector3();
  let count = 0;
  for (let i = 0; i < layer.count; i += 1) {
    if (!wanted.has(layer.bone[i])) continue;
    centre.x += layer.positions[i * 3];
    centre.y += layer.positions[i * 3 + 1];
    centre.z += layer.positions[i * 3 + 2];
    count += 1;
  }
  if (!count) { status('这根骨没有顶点'); return; }
  centre.divideScalar(count);
  const offset = camera.position.clone().sub(orbit).setLength(0.55);
  orbit.copy(centre);
  camera.position.copy(centre).add(offset);
  camera.lookAt(orbit);
  status(`已聚焦 ${boneLabelOf(target)}（${count} 个顶点）`);
};

$('v-home').onclick = () => { resetCamera(); status('已重置视角'); };

addEventListener('keydown', (event) => {
  if (event.target.tagName === 'INPUT') return;
  if (event.key === '[') selectTarget(state.active - 1);
  else if (event.key === ']') selectTarget(state.active + 1);
  else if (event.key === 'r' || event.key === 'R') {
    resetCamera(); status('已重置视角');
  } else if (event.key === 'f' || event.key === 'F') $('v-focus').click();
  else if (event.key === 'q' || event.key === 'Q') rollCamera(+1);
  else if (event.key === 'e' || event.key === 'E') rollCamera(-1);
  else if (event.key === 'c' || event.key === 'C') $('t-clip').click();
  else if (event.key === 'i' || event.key === 'I') $('t-isolate').click();
  else if (event.key === 'w' || event.key === 'W') $('t-wire').click();
});

load().then(render);
