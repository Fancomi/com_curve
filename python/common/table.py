"""CSV 表格的写与读。四份手写的 f-string 拼接收敛到这里。

写侧统一 nan -> 空串：非有限值写成 `nan` 字面量的话，Excel 与 pandas 的默认解析
各有一套猜法，而空串在两边都是缺失值。dive 那条线本来就有大段无效帧。
"""
import csv

import numpy as np


def cell(value, digits=3):
    """一个数值格子。非有限值给空串。"""
    return "" if not np.isfinite(value) else f"{value:.{digits}f}"


def write_table(path, columns, digits=3):
    """`columns` 是 [(列名, 一维数组)]，长度必须一致。返回写出的路径。"""
    names = [name for name, _values in columns]
    arrays = [np.asarray(values) for _name, values in columns]
    lengths = {len(a) for a in arrays}
    if len(lengths) != 1:
        raise ValueError(f"列长不一致：{dict(zip(names, map(len, arrays)))}")
    lines = [",".join(names)]
    for row in range(arrays[0].shape[0]):
        lines.append(",".join(cell(a[row], digits) for a in arrays))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def vector_columns(prefix, track, suffix, axes="xyz"):
    """把 (N, 3) 拆成三列 [(f"{prefix}_{axis}_{suffix}", 列)]。"""
    return [(f"{prefix}_{axis}_{suffix}", track[:, k])
            for k, axis in enumerate(axes)]


def read_table(path):
    """CSV -> {列名: float 数组}。空串读成 nan。"""
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} 是空表")
    return {key: np.array([float(row[key]) if row[key] else np.nan
                           for row in rows])
            for key in rows[0]}
