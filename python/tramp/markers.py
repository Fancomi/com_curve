"""把重心曲线写成带动画的 FBX：每条曲线一个 null + 一条 FbxLine 轨迹。

三维产物比图表方便：null 与源文件同帧号、同时基，拖进 Maya/Blender 就能和人体逐帧
对上；FbxLine 让整条弧线在视口里静态可见，不用播动画。两者位置逐帧重合，可以互相
校验。
"""
import numpy as np

from . import scene as F


def write(path, frames, tracks, time_mode, scene=None, manager=None, size=12.0):
    """`tracks` 是 {名字: (N, 3)}。给了 `scene` 就并进那个场景，否则新建一个。

    null 的关键帧写成 cubic：线性会让曲线在时间轴上看着有折角，而这条曲线本身是
    平滑的抛物线族，折角全是插值造成的假象。

    新场景必须显式继承源场景的 time mode。FbxScene 默认是 eDefaultMode（30 fps），
    而蹦床这个文件是 ePAL（25）；帧号照抄但时基不同，导出的弧线会以 1.2 倍速播放，
    逐帧比对时还发现不了——两边帧号完全对得上。
    """
    import fbx

    own = scene is None
    if own:
        manager = F.open_manager()
        scene = fbx.FbxScene.Create(manager, "com_curve")
        settings = scene.GetGlobalSettings()
        settings.SetSystemUnit(fbx.FbxSystemUnit.cm)
        settings.SetAxisSystem(fbx.FbxAxisSystem.MayaYUp)
        settings.SetTimeMode(time_mode)
        stack = fbx.FbxAnimStack.Create(scene, "Take 001")
        layer = fbx.FbxAnimLayer.Create(scene, "Base Layer")
        stack.AddMember(layer)
        scene.SetCurrentAnimationStack(stack)
        span = fbx.FbxTimeSpan(F.time_at(frames[0], time_mode),
                               F.time_at(frames[-1], time_mode))
        settings.SetTimelineDefaultTimeSpan(span)
        stack.SetLocalTimeSpan(span)
    else:
        time_mode = scene.GetGlobalSettings().GetTimeMode()
        layer = F.base_layer(scene)

    group = fbx.FbxNode.Create(scene, "COM_Curves")
    scene.GetRootNode().AddChild(group)
    cubic = fbx.FbxAnimCurveDef.EInterpolationType.eInterpolationCubic

    for name, track in tracks.items():
        marker = fbx.FbxNull.Create(scene, f"COM_{name}_shape")
        marker.Size.Set(size)
        node = fbx.FbxNode.Create(scene, f"COM_{name}")
        node.SetNodeAttribute(marker)
        group.AddChild(node)
        for axis, column in (("X", 0), ("Y", 1), ("Z", 2)):
            curve = node.LclTranslation.GetCurve(layer, axis, True)
            curve.KeyModifyBegin()
            for index, frame in enumerate(frames):
                key = curve.KeyAdd(F.time_at(frame, time_mode))[0]
                curve.KeySetValue(key, float(track[index, column]))
                curve.KeySetInterpolation(key, cubic)
            curve.KeyModifyEnd()

        line = fbx.FbxLine.Create(scene, f"COM_{name}_trail_shape")
        line.InitControlPoints(len(track))
        for index, point in enumerate(np.asarray(track, dtype=float)):
            line.SetControlPointAt(fbx.FbxVector4(*point), index)
            line.AddPointIndex(index)
        trail = fbx.FbxNode.Create(scene, f"COM_{name}_trail")
        trail.SetNodeAttribute(line)
        group.AddChild(trail)

    F.save(manager, scene, path)
    if own:
        manager.Destroy()
    return path
