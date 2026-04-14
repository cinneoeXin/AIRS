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

# ========== Configuration: Change to your file path. ==========
# support .xlsx / .csv / .parquet
TRAIN_PATHS = [
    "E:/USA/AIRS/AIRS WEEK/WEEK10/Network_dataset_1.xlsx",            # Flow Class
    "E:/USA/AIRS/AIRS WEEK/WEEK10/matched_columns_file_TII_B.xlsx",   # Flow Class
]
VAL_PATHS = [
    "E:/USA/AIRS/AIRS WEEK/WEEK10/Network_dataset_1.xlsx",
    "E:/USA/AIRS/AIRS WEEK/WEEK10/matched_columns_file_TII_B.xlsx",
]
TEST_PATHS = [
    "E:/USA/AIRS/AIRS WEEK/WEEK10/VPN_email_classified.xlsx",         # Package-level（include Traffic_Type）
]

OUT_DIR = "./harmonized/scenario_b"
WRITE_PARQUET = True     # True: Write Parquet; False: Write CSV

# ========== Tag Recognition Settings ==========
# Perform explicit mapping for your three datasets (based on filenames, excluding extensions).
# If your filename differs, please change the key to your actual basename.
LABEL_MAP = {
    "VPN_email_classified": "traffic_type",          # Traffic_Type → traffic_type
    "Network_dataset_1": "type",                     # type → type
    "matched_columns_file_TII_B": "label",          # Label → label
}

# Expand Candidate Names (Attempted both before and after standardization)
LABEL_CANDIDATES_RAW = [
    "label","Label","LABEL","y","Y","type","Type","Traffic_Type"
]
# Smart Keywords (Matched after Standardization)
LABEL_KEYWORDS_RE = re.compile(
    r"(label|class|attack|category|malicious|benign|target|type)$",
    re.I
)

# Obvious Non-Feature Columns (Discarded Upon Appearance) — Note: These refer to the names **after standardization**.
NON_FEATURE_CANDS = {
    "ts","timestamp","time","flow_start_ts",
    "src","dst","src_ip","dst_ip","source","destination",
    "service","dns_query","ssl_subject","ssl_issuer",
    "http_method","http_uri","http_referrer","http_user_agent",
    "http_orig_mime_types","http_resp_mime_types",
    "weird_name","weird_addl","info","no","_time_s"
}

# ========== Semantic Normalization: Rules and Synonyms ==========
CANON_SYNONYMS: List[Tuple[str, str]] = [
    # duration
    (r"^(flow_)?duration$", "duration"),
    (r"^flow[_ ]?duration$", "duration"),

    # Directory Entry Size (Retain Original Name)
    (r"^(fwd_)?segment_size_avg$", "fwd_segment_size_avg"),
    (r"^bwd[_ ]?segment_size_avg$", "bwd_segment_size_avg"),

    # Packet Length / Byte Statistics → pkt_size_*
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

    # active/idle Interval Statistics
    (r"^active_(mean)$", "active_mean"),
    (r"^active_(std)$",  "active_std"),
    (r"^active_(min)$",  "active_min"),
    (r"^active_(max)$",  "active_max"),
    (r"^idle_(mean)$",   "idle_mean"),
    (r"^idle_(std)$",    "idle_std"),
    (r"^idle_(min)$",    "idle_min"),
    (r"^idle_(max)$",    "idle_max"),

    # Port/Protocol
    (r"^(src[_ ]?port)$", "src_port"),
    (r"^(dst[_ ]?port)$", "dst_port"),
    (r"^(protocol|proto)$", "protocol"),
]

# ========== Utility Functions ==========
def _strip_invisible(s: str) -> str:
    # Remove BOM, zero-width characters, non-breaking spaces, etc.
    return (
        s.replace("\u200b", "")
         .replace("\ufeff", "")
         .replace("\xa0", " ")
    )

def normalize_name(name: str) -> str:
    # Standardize Column Names: Remove invisible characters; convert to lowercase; replace spaces, hyphens, and periods with underscores; and remove redundant underscores.
    s = _strip_invisible(str(name))
    s = s.strip().lower()
    s = re.sub(r"[ \t\.\-\/]+", "_", s)
    s = re.sub(r"_+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)  # Conservative: Remove Strange Symbols
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
    # Fewer categories; functions more like tags (adjustable threshold).
    if nuniq <= 20:
        if (pd.api.types.is_categorical_dtype(s) or
            pd.api.types.is_string_dtype(s) or
            pd.api.types.is_integer_dtype(s)):
            return True
    return False

def detect_label_col(df: pd.DataFrame, name: str|None=None) -> str:
    """
    Label Column Identification Priority:
    1) Explicit Mapping: `LABEL_MAP[name]`
    2) Candidate Names (Normalized)
    3) Keyword Match + Structural Heuristics
    4) Table-wide Fallback: Select a column that "looks like a label"
    """
    df_norm = normalize_columns(df)
    cols = list(df_norm.columns)

    # 1) Explicit Mapping
    if name and name in LABEL_MAP:
        mapped = LABEL_MAP[name]
        if mapped in cols:
            return mapped
        else:
            # Diagnostic Information
            raise ValueError(
                f"[{name}] Explicitly map the specified label column '{mapped}' not found"
                f"\nExisting (Standardized) Column Names (Max 50): {cols[:50]}"
            )

    # 2) Candidate Name (Original candidates are standardized prior to matching)
    candidates_norm = [normalize_name(c) for c in LABEL_CANDIDATES_RAW]
    for c in candidates_norm:
        if c in cols:
            return c

    # 3) Keywords + Pattern Recognition
    kw_hits = [c for c in cols if LABEL_KEYWORDS_RE.search(c)]
    kw_hits = [c for c in kw_hits if looks_like_label(df_norm[c])]
    if kw_hits:
        kw_hits.sort(key=lambda c: df_norm[c].nunique(dropna=True))
        return kw_hits[0]

    # 4) Full-Table Catch-all
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

# ========== Core Processing ==========
def harmonize_one(path: str, name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (df_canon, colmap)
    - df_canon: Table with standardized column names (numerical features + label only), not yet trimmed to the intersection.
    - colmap: Field mapping: Original -> Standardized -> Canonical.
    """
    raw = load_any(path)
    raw_cols = list(raw.columns)

    # Standardize Column Names
    df = normalize_columns(raw)
    norm_cols = list(df.columns)

    # Identify Label Column (Pass filename as dataset name)
    label_col_norm = detect_label_col(df, name=name)

    # Save Diagnostic: Original Label Column Name
    try:
        idx = norm_cols.index(label_col_norm)
        original_label = raw_cols[idx]
    except Exception:
        original_label = label_col_norm  # 兜底

    # Apply synonym mapping + standardize the label column name to 'label'
    mapped_cols = []
    for c in df.columns:
        if c == label_col_norm:
            mapped_cols.append("label")
        else:
            mapped_cols.append(apply_synonyms(c))
    df.columns = mapped_cols

    # Generate Optional Derived Columns
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

    # Exclusion of Non-Feature Columns (After Standardization）
    drop_cands = set(NON_FEATURE_CANDS)
    feature_cols = [c for c in df.columns if c not in drop_cands and c != "label"]

    # Retain only numerical features (force-convert others to numerical types; if conversion fails, set to NaN, then to 0).
    df_feat = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df_feat["label"] = df["label"].copy()

    # Construct Mapping Table: Original Column -> Standardization -> Normalization
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

# ========== Main Process ==========
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

    # Consolidation Table by Table
    for role, path in all_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"文件不存在: {path}")
        name = os.path.splitext(os.path.basename(path))[0]  # As dataset name
        df_canon, colmap = harmonize_one(path, name)
        harmonized_by_role[role].append((name, df_canon, colmap))
        all_colmaps.append(colmap)

    # Calculate Intersection（train+val+test）
    dfs_for_common = [d for role in harmonized_by_role for (_, d, _) in harmonized_by_role[role]]
    common_feats = intersect_features(dfs_for_common)

    # output common_features.txt
    with open(os.path.join(OUT_DIR, "common_features.txt"), "w", encoding="utf-8") as f:
        for c in common_feats:
            f.write(c + "\n")

    # Crop each dataset to their intersection and write them out.
    for role in ["train","val","test"]:
        for name, df_can, _ in harmonized_by_role[role]:
            keep_cols = common_feats + ["label"]
            # The intersection may be empty, and exporting is permitted (containing only the label); if a minimum of N features is required, an assertion can be added.
            out = df_can[keep_cols].copy()
            out_path = os.path.join(
                OUT_DIR, f"{name}_harmonized." + ("parquet" if WRITE_PARQUET else "csv")
            )
            write_table(out, out_path)

    # Summary Field Mapping
    colmap_all = pd.concat(all_colmaps, ignore_index=True)
    colmap_all_path = os.path.join(OUT_DIR, "all_column_mappings.xlsx")
    colmap_all.to_excel(colmap_all_path, index=False)

    print("Scenario B harmonization finished.")
    print("Common features count:", len(common_feats))
    print("Output dir:", OUT_DIR)

if __name__ == "__main__":
    main()
