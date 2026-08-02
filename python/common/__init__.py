"""跨两条 pipeline 共用的底座：纯 numpy / cv2，不 import fbx，不认识业务口径。

`geometry` 网格几何，`signal` 曲线与抛体，`table` CSV 读写，`raster` 软件光栅器，
`plotstyle` matplotlib 中文字体与配色。这一层没有"重心"或"腾空"的概念——
那些属于 `tramp` / `dive`。
"""
