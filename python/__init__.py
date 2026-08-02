"""com_curve 的 Python 包。两条数据线 + 一层共用底座。

    common/  纯 numpy / cv2 的底座：几何、信号、表格、光栅器、绘图样式
    tramp/   蹦床 FBX 线（BengChuang_01.fbx）
    dive/    跳水 DIVF replay 线（*_replay_frames.bin.br）

入口一律 `python -m python.tramp <步骤>` / `python -m python.dive <步骤>`，
或者用 `scripts/` 下的 sh 包一层。
"""
