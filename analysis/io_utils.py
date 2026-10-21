"""通用 CSV 读取：编码、分隔符、表头都不写死。

设计原则（来自审查意见）：
- 不用固定列名判断“读取成功”，列名变了也必须能读；
- 编码按可解码性 + 可打印性判断；
- 统一先按字符串读入，保留订单号前导零，数值/日期在口径层再转换。
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pandas as pd

ENCODINGS = ("utf-8-sig", "utf-8", "gb18030")


def _decode(raw: bytes) -> tuple[str, str, bool]:
    """按优先级解码；返回 (文本, 编码, 是否降级兜底)。"""
    for enc in ENCODINGS:
        try:
            text = raw.decode(enc)
        except UnicodeDecodeError:
            continue
        if "�" in text or "\x00" in text:
            continue
        return text, enc, False
    # 兜底：不因编码问题中断，但显式标记，结果里能看到 decoding_fallback=true
    return raw.decode("latin-1", errors="replace"), "latin-1", True


def _sniff_delimiter(text: str) -> str:
    sample = text[:8192]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        # 用首行各候选分隔符出现次数兜底
        first = sample.splitlines()[0] if sample.splitlines() else ""
        counts = {d: first.count(d) for d in ",;\t|"}
        best = max(counts, key=counts.get)
        return best if counts[best] > 0 else ","


def read_table(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """读取任意 CSV，返回 DataFrame 与读取元信息。"""
    path = Path(path)
    raw = path.read_bytes()
    text, encoding, fallback = _decode(raw)
    delimiter = _sniff_delimiter(text)

    read_kwargs = dict(
        sep=delimiter,
        dtype=str,
        keep_default_na=True,
        na_values=["", "NA", "N/A", "NaN", "nan", "null", "NULL"],
        engine="c",
    )
    try:
        df = pd.read_csv(io.StringIO(text), **read_kwargs)
        engine = "c"
    except Exception:
        # 畸形引号/换行等 C 引擎读不了的文件，退回 python 引擎
        read_kwargs["engine"] = "python"
        df = pd.read_csv(io.StringIO(text), **read_kwargs)
        engine = "python"

    # 列名去空白；完全空白的列名补编号，避免后续 groupby 报错
    columns = []
    for i, col in enumerate(df.columns):
        name = str(col).strip()
        if not name or name.lower().startswith("unnamed"):
            name = f"未命名列_{i + 1}"
        columns.append(name)
    df.columns = columns

    meta = {
        "file": path.name,
        "encoding": encoding,
        "delimiter": delimiter,
        "decoding_fallback": fallback,
        "engine": engine,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "column_names": columns,
    }
    return df, meta
