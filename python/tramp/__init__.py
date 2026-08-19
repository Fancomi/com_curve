"""tramp —— 蹦床 FBX 那条线：读 `BengChuang_01.fbx`，算重心，反算根骨骼。

    scene     FBX SDK 的唯一封装层（整个项目只有这里 import fbx）
    segments  14 段人体模型的纯数据（Dempster 表、段定义、关节连线质心）
    bodies    绑定姿态编译出的 14 段刚体（体素化 + 质量 + 冻结质心）
    curves    五种重心口径的逐帧采样。**唯一知道"重心怎么算"的地方**
    markers   把曲线写成带动画的 FBX null + FbxLine 轨迹
    retarget  在"某条曲线走完美抛物线"前提下反算根骨骼
    verify    独立校验（解析体、蒙特卡洛、体段恒等式、反算产物）

入口在 `scripts/`，不在这里。
"""
