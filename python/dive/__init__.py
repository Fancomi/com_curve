"""dive —— 跳水 DIVF replay 那条线：读 `*_replay_frames.bin.br`，算重心，叠回源视频。

    divf    二进制格式解析（brotli 解压 + 逐帧记录）
    align   固定深度平面对齐与尺度标定。**这条线所有"为什么"都在这里**
    curves  两种重心口径的逐帧采样
    plots   图表
    overlay 叠回源视频的二维渲染件（曲线板、HUD）

不需要 FBX SDK。入口在 `scripts/`。
"""
