# -*- coding: utf-8 -*-
"""
Harmonize for Scenario B (cross-domain generalization)
- Train/Val on dataset(s) A (and/or B), Test on dataset C
- Produce: per-dataset harmonized tables with identical common features (+label)

Outputs (under ./harmonized/scenario_b/):
- <name>_harmonized.parquet (or .csv)
- all_column_mappings.xlsx  (original -> canonical mapping, and notes)
- common_features.txt
"""

import os
import re
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# ========== 配置：修改为你的文件路径 ==========
# 支持 .xlsx / .csv / .parquet
TRAIN_PATHS = [
    "E:/USA/AIRS/AIRS WEEK/WEEK10/Network_dataset_1.xlsx",            # 流级
    "E:/USA/AIRS/AIRS WEEK/WEEK10/matched_columns_file_TII_B.xlsx",   # 流级
]
VAL_PATHS = [
    "E:/USA/AIRS/AIRS WEEK/WEEK10/Network_dataset_1.xlsx",
    "E:/USA/AIRS/AIRS WEEK/WEEK10/matched_columns_file_TII_B.xlsx",
]
TEST_PATHS = [
    "E:/USA/AIRS/AIRS WEEK/WEEK10/VPN_email_classified.xlsx",         # 包级（含 Traffic_Type）
]

OUT_DIR = "./harmonized/scenario_b"
WRITE_PARQUET = True     # True: 写 parquet；False: 写 csv

# ========== 标签识别设置 ==========
# 针对你的三个数据集（按文件名不含扩展名）做显式映射
# 若你的文件名不同，请把 key 改为你的实际 basename
LABEL_MAP = {
    "VPN_email_classified": "traffic_type",          # Traffic_Type → traffic_type
    "Network_dataset_1": "type",                     # type → type
    "matched_columns_file_TII_B": "label",          # Label → label
}

# 扩展候选名（标准化前后都会尝试）
LABEL_CANDIDATES_RAW = [
    "label","Label","LABEL","y","Y","type","Type","Traffic_Type"
]
# 智能关键字（标准化后匹配）
LABEL_KEYWORDS_RE = re.compile(
    r"(label|class|attack|category|malicious|benign|target|type)$",
    re.I
)

# 明显的非特征列（出现即剔除） —— 注意为**标准化后**的名字
NON_FEATURE_CANDS = {
    "ts","timestamp","time","flow_start_ts",
    "src","dst","src_ip","dst_ip","source","destination",
    "service","dns_query","ssl_subject","ssl_issuer",
    "http_method","http_uri","http_referrer","http_user_agent",
    "http_orig_mime_types","http_resp_mime_types",
    "weird_name","weird_addl","info","no","_time_s"
}

# ========== 语义归一：规则与同义词 ==========
CANON_SYNONYMS: List[Tuple[str, str]] = [
    # duration
    (r"^(flow_)?duration$", "duration"),
    (r"^flow[_ ]?duration$", "duration"),

    # 方向段大小（保留原名）
    (r"^(fwd_)?segment_size_avg$", "fwd_segment_size_avg"),
    (r"^bwd[_ ]?segment_size_avg$", "bwd_segment_size_avg"),

    # 包长/字节统计 → pkt_size_*
    (r"^(len|length|bytes)_(mean)$", "pkt_size_mean"),
    (r"^(len|length|bytes)_(std)$", "pkt_size_std"),
    (r"^(len|length|bytes)_(min)$", "pkt_size_min"),
    (r"^(len|length|bytes)_(max)$", "pkt_size_max"),
    (r"^(len|length|bytes)_(avg|average)$", "pkt_size_mean"),

    # inter-arrival time
    (r"^(iat|inter[_ ]?arrival[_ ]?time)_(mean)$", "iat_mean"),
    (r"^(iat|inter[_ ]?arrival[_ ]?time)_(std)$", "iat_std"),
    (r"^(iat|inter[_ ]?arrival[_ ]?time)_(min)$", "iat_min"),
    (r"^(iat|inter[_ ]?arrival[_ ]?time)_(max)$", "iat_max"),

    # totals
    (r"^(bytes_total|total_bytes)$", "bytes_total"),
    (r"^(pkt_cnt|packet_count|packets)$", "packet_count"),

    # active/idle 区间统计
    (r"^active_(mean)$", "active_mean"),
    (r"^active_(std)$",  "active_std"),
    (r"^active_(min)$",  "active_min"),
    (r"^active_(max)$",  "active_max"),
    (r"^idle_(mean)$",   "idle_mean"),
    (r"^idle_(std)$",    "idle_std"),
    (r"^idle_(min)$",    "idle_min"),
    (r"^idle_(max)$",    "idle_max"),

    # 端口/协议
    (r"^(src[_ ]?port)$", "src_port"),
    (r"^(dst[_ ]?port)$", "dst_port"),
    (r"^(protocol|proto)$", "protocol"),
]

# ========== 工具函数 ==========
def _strip_invisible(s: str) -> str:
    # 去掉 BOM/零宽/不间断空格等
    return (
        s.replace("\u200b", "")
         .replace("\ufeff", "")
         .replace("\xa0", " ")
    )

def normalize_name(name: str) -> str:
    # 标准化列名：去不可见字符、小写、空白/连字符/点→下划线、去多余下划线
    s = _strip_invisible(str(name))
    s = s.strip().lower()
    s = re.sub(r"[ \t\.\-\/]+", "_", s)
    s = re.sub(r"_+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)  # 保守：移除奇异符号
    return s

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: normalize_name(c) for c in df.columns})

def apply_synonyms(n: str) -> str:
    for pat, canon in CANON_SYNONYMS:
        if re.fullmatch(pat, n):
            return canon
    return n

def looks_like_label(s: pd.Series) -> bool:
    nuniq = s.nunique(dropna=True)
    if nuniq == 0:
        return False
    # 类别较少更像标签（阈值可调）
    if nuniq <= 20:
        if (pd.api.types.is_categorical_dtype(s) or
            pd.api.types.is_string_dtype(s) or
            pd.api.types.is_integer_dtype(s)):
            return True
    return False

def detect_label_col(df: pd.DataFrame, name: str|None=None) -> str:
    """
    标签列识别优先级：
    1) 显式映射 LABEL_MAP[name]
    2) 候选名（标准化）
    3) 关键字命中 + 形态判断
    4) 全表兜底：挑一个“像标签”的列
    """
    df_norm = normalize_columns(df)
    cols = list(df_norm.columns)

    # 1) 显式映射
    if name and name in LABEL_MAP:
        mapped = LABEL_MAP[name]
        if mapped in cols:
            return mapped
        else:
            # 诊断信息
            raise ValueError(
                f"[{name}] 显式映射指定的标签列 '{mapped}' 未找到。"
                f"\n现有（标准化后）列名(最多50): {cols[:50]}"
            )

    # 2) 候选名（原始候选做标准化后再匹配）
    candidates_norm = [normalize_name(c) for c in LABEL_CANDIDATES_RAW]
    for c in candidates_norm:
        if c in cols:
            return c

    # 3) 关键字 + 形态判断
    kw_hits = [c for c in cols if LABEL_KEYWORDS_RE.search(c)]
    kw_hits = [c for c in kw_hits if looks_like_label(df_norm[c])]
    if kw_hits:
        kw_hits.sort(key=lambda c: df_norm[c].nunique(dropna=True))
        return kw_hits[0]

    # 4) 全表兜底
    cands = [c for c in cols if looks_like_label(df_norm[c])]
    if cands:
        cands.sort(key=lambda c: df_norm[c].nunique(dropna=True))
        return cands[0]

    raise ValueError(
        "Label column not found."
        f"\nTried (normalized): {', '.join(candidates_norm)}"
        f"\nColumns seen (normalized, up to 50): {cols[:50]}"
    )

def load_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    elif ext == ".csv":
        return pd.read_csv(path)
    elif ext == ".parquet":
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

# ========== 核心处理 ==========
def harmonize_one(path: str, name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    返回 (df_canon, colmap)
    - df_canon：规范列名的表（仅数值特征 + label），尚未裁剪到交集
    - colmap：原始 -> 标准化 -> 规范 的字段映射
    """
    raw = load_any(path)
    raw_cols = list(raw.columns)

    # 标准化列名
    df = normalize_columns(raw)
    norm_cols = list(df.columns)

    # 识别标签列（传入文件名作为数据集名）
    label_col_norm = detect_label_col(df, name=name)

    # 保存诊断：原始标签列名字
    try:
        idx = norm_cols.index(label_col_norm)
        original_label = raw_cols[idx]
    except Exception:
        original_label = label_col_norm  # 兜底

    # 应用同义词映射 + 标签列标准化名为 'label'
    mapped_cols = []
    for c in df.columns:
        if c == label_col_norm:
            mapped_cols.append("label")
        else:
            mapped_cols.append(apply_synonyms(c))
    df.columns = mapped_cols

    # 生成可选派生列
    if set(["subflow_fwd_packets","subflow_bwd_packets"]).issubset(df.columns):
        df["packet_count"] = (
            pd.to_numeric(df["subflow_fwd_packets"], errors="coerce").fillna(0)
            + pd.to_numeric(df["subflow_bwd_packets"], errors="coerce").fillna(0)
        )

    if set(["subflow_fwd_bytes","subflow_bwd_bytes"]).issubset(df.columns):
        df["bytes_total"] = (
            pd.to_numeric(df["subflow_fwd_bytes"], errors="coerce").fillna(0)
            + pd.to_numeric(df["subflow_bwd_bytes"], errors="coerce").fillna(0)
        )

    # 非特征列剔除（标准化后）
    drop_cands = set(NON_FEATURE_CANDS)
    feature_cols = [c for c in df.columns if c not in drop_cands and c != "label"]

    # 仅保留数值型特征（其他强转为数值，失败→NaN→0）
    df_feat = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df_feat["label"] = df["label"].copy()

    # 构建映射表：原始列 -> 标准化 -> 规范
    colmap = pd.DataFrame({
        "raw_column": raw_cols,
        "normalized": norm_cols,
        "canonical": mapped_cols
    })
    colmap.loc[colmap["canonical"]=="label", "note"] = f"Label (from {original_label})"
    colmap["dataset"] = name

    return df_feat, colmap

def intersect_features(dfs: List[pd.DataFrame]) -> List[str]:
    sets = []
    for d in dfs:
        sets.append(set([c for c in d.columns if c != "label"]))
    common = set.intersection(*sets) if sets else set()
    return sorted(list(common))

def write_table(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if WRITE_PARQUET:
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)

# ========== 主流程 ==========
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    all_paths: List[Tuple[str, str]] = []
    for p in TRAIN_PATHS: all_paths.append(("train", p))
    for p in VAL_PATHS:   all_paths.append(("val", p))
    for p in TEST_PATHS:  all_paths.append(("test", p))

    harmonized_by_role: Dict[str, List[Tuple[str, pd.DataFrame, pd.DataFrame]]] = {
        "train":[], "val":[], "test":[]
    }
    all_colmaps = []

    # 逐表归一
    for role, path in all_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"文件不存在: {path}")
        name = os.path.splitext(os.path.basename(path))[0]  # 作为数据集名
        df_canon, colmap = harmonize_one(path, name)
        harmonized_by_role[role].append((name, df_canon, colmap))
        all_colmaps.append(colmap)

    # 计算交集（train+val+test）
    dfs_for_common = [d for role in harmonized_by_role for (_, d, _) in harmonized_by_role[role]]
    common_feats = intersect_features(dfs_for_common)

    # 输出 common_features.txt
    with open(os.path.join(OUT_DIR, "common_features.txt"), "w", encoding="utf-8") as f:
        for c in common_feats:
            f.write(c + "\n")

    # 各数据集裁剪到交集并写出
    for role in ["train","val","test"]:
        for name, df_can, _ in harmonized_by_role[role]:
            keep_cols = common_feats + ["label"]
            # 交集可能为空，也允许导出（只含 label）；如需强制至少 N 个特征，可加断言
            out = df_can[keep_cols].copy()
            out_path = os.path.join(
                OUT_DIR, f"{name}_harmonized." + ("parquet" if WRITE_PARQUET else "csv")
            )
            write_table(out, out_path)

    # 汇总字段映射
    colmap_all = pd.concat(all_colmaps, ignore_index=True)
    colmap_all_path = os.path.join(OUT_DIR, "all_column_mappings.xlsx")
    colmap_all.to_excel(colmap_all_path, index=False)

    print("Scenario B harmonization finished.")
    print("Common features count:", len(common_feats))
    print("Output dir:", OUT_DIR)

if __name__ == "__main__":
    main()
