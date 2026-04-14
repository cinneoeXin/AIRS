# -*- coding: utf-8 -*-
"""
Preprocess for Dataset 2: VPN_email_classified.xlsx (packet-level)
Outputs:
- ./processed/dataset2/ml_arrays.npz         (X_train/y_train/X_val/y_val/X_test/y_test/feature_names)
- ./processed/dataset2/train.xlsx, val.xlsx, test.xlsx
- ./processed/dataset2/scaler.pkl
- ./processed/dataset2/dl_arrays_seq.npz     (X_train_seq/y_train/... channels/seq_len)
"""

import os
import pickle
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============== 配置区域 ==============
in_path = "E:/USA/AIRS/AIRS WEEK/WEEK10/VPN_email_classified.xlsx"  # ← 改成你的文件路径
out_dir = "./processed/dataset2"
os.makedirs(out_dir, exist_ok=True)

# 列名映射（与你的数据表头一致）
COL_NO = "No."
COL_TIME = "Time"
COL_SRC = "Source"
COL_DST = "Destination"
COL_PROTO = "Protocol"
COL_LEN = "Length"
COL_INFO = "Info"           # 不参与特征
COL_LABEL = "Traffic_Type"  # 标签列（如 Normal / Suspect）

# 流切分超时（秒）
FLOW_TIMEOUT_S = 120.0

# DL 序列参数
SEQ_LEN = 128
MIN_PKTS_FOR_DL = 3
DL_CHANNELS = ["Length", "delta_t", "direction"]  # 三通道：包长、间隔、方向

# 标签映射（大小写不敏感）
# 默认：Normal/Benign -> 0，其他 -> 1
def encode_label(v: str) -> int:
    if v is None:
        return 1
    s = str(v).strip().lower()
    if s in {"normal", "benign"}:
        return 0
    return 1


# ============== 工具函数 ==============
def canonical_pair(src: str, dst: str) -> Tuple[str, str, int]:
    """
    把 (src, dst) 变成无向对：按字典序排序，返回 (a, b, sign)
    sign = +1 表示原始方向 src==a (正向)，-1 表示原始方向为反向
    """
    a, b = sorted([str(src), str(dst)])
    sign = +1 if str(src) == a else -1
    return a, b, sign


def build_flow_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    依据 (canonical {Source,Destination} + Protocol) + 超时 切分 flow_id
    要求 df 已按 key 和时间升序排序
    """
    flow_ids = []
    last_time_by_key: Dict[Tuple[str, str, str], float] = {}
    flow_index_by_key: Dict[Tuple[str, str, str], int] = {}

    for idx, row in df.iterrows():
        key = (row["_src_can"], row["_dst_can"], row[COL_PROTO])
        t = row["_time_s"]
        if key not in last_time_by_key:
            last_time_by_key[key] = t
            flow_index_by_key[key] = 0
        # 超时 -> 新流
        if t - last_time_by_key[key] > FLOW_TIMEOUT_S:
            flow_index_by_key[key] += 1
        last_time_by_key[key] = t
        flow_ids.append(f"{key[0]}|{key[1]}|{key[2]}|{flow_index_by_key[key]}")
    df["flow_id"] = flow_ids
    return df


# ============== 读入与基础加工 ==============
print("[Dataset2] Loading Excel …")
df = pd.read_excel(in_path)

# 基本字段检查
required_cols = [COL_TIME, COL_SRC, COL_DST, COL_PROTO, COL_LEN, COL_LABEL]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# 时间转秒（已是相对时间，直接 float）
df["_time_s"] = pd.to_numeric(df[COL_TIME], errors="coerce").astype(float).fillna(0.0)
# 包长
df["_len"] = pd.to_numeric(df[COL_LEN], errors="coerce").astype(float).fillna(0.0)

# 无向对 + 方向
src_can, dst_can, sign_list = [], [], []
for src, dst in zip(df[COL_SRC].astype(str), df[COL_DST].astype(str)):
    a, b, s = canonical_pair(src, dst)
    src_can.append(a); dst_can.append(b); sign_list.append(s)
df["_src_can"] = src_can
df["_dst_can"] = dst_can
df["_dir_sign"] = np.array(sign_list, dtype=float)

# 排序并切分 flow
df = df.sort_values(by=["_src_can", "_dst_can", COL_PROTO, "_time_s"], kind="mergesort").reset_index(drop=True)
df = build_flow_ids(df)

# 计算 Δt（同 flow 内）
df["delta_t"] = df.groupby("flow_id")["_time_s"].diff().fillna(0.0)

# 统一标签二值化
df["_y"] = df[COL_LABEL].apply(encode_label).astype(int)

# ============== ML：生成流级统计特征 ==============
print("[Dataset2] Building flow-level features for ML …")
grp = df.groupby("flow_id")
len_mean = grp["_len"].mean()
len_std  = grp["_len"].std().fillna(0.0)
len_min  = grp["_len"].min()
len_max  = grp["_len"].max()
iat_mean = grp["delta_t"].mean()
iat_std  = grp["delta_t"].std().fillna(0.0)
iat_min  = grp["delta_t"].min()
iat_max  = grp["delta_t"].max()
pkt_cnt  = grp.size()
bytes_total = grp["_len"].sum()

# 流持续时间
t_start = grp["_time_s"].min()
t_end   = grp["_time_s"].max()
duration = t_end - t_start

# 正/反向包比例（基于无向对里的 sign）
pos_ratio = grp["_dir_sign"].apply(lambda s: (s > 0).mean() if len(s) else 0.5)

# 多数表决标签
y_flow = grp["_y"].agg(lambda x: x.value_counts().index[0])

flows_df = pd.DataFrame({
    "flow_id": len_mean.index,
    "pkt_cnt": pkt_cnt.values,
    "bytes_total": bytes_total.values,
    "len_mean": len_mean.values,
    "len_std": len_std.values,
    "len_min": len_min.values,
    "len_max": len_max.values,
    "iat_mean": iat_mean.values,
    "iat_std": iat_std.values,
    "iat_min": iat_min.values,
    "iat_max": iat_max.values,
    "duration": duration.values,
    "dir_ratio_pos": pos_ratio.values,
    "y": y_flow.values
})

# 选择 ML 特征列
ml_feature_cols = [
    "pkt_cnt","bytes_total","len_mean","len_std","len_min","len_max",
    "iat_mean","iat_std","iat_min","iat_max","duration","dir_ratio_pos"
]

X_ml = flows_df[ml_feature_cols].astype(float)
y_ml = flows_df["y"].astype(int).to_numpy()

# 标准化 & 切分
scaler = StandardScaler()
X_ml_scaled = scaler.fit_transform(X_ml)

with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

X_tr, X_tmp, y_tr, y_tmp = train_test_split(
    X_ml_scaled, y_ml, test_size=0.30, random_state=42, stratify=y_ml
)
X_va, X_te, y_va, y_te = train_test_split(
    X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
)

# 保存 ML .npz
np.savez_compressed(
    os.path.join(out_dir, "ml_arrays.npz"),
    X_train=X_tr, y_train=y_tr,
    X_val=X_va, y_val=y_va,
    X_test=X_te, y_test=y_te,
    feature_names=np.array(ml_feature_cols)
)

# 另存 ML .xlsx（便于查看）
pd.DataFrame(X_tr, columns=ml_feature_cols).assign(label=y_tr).to_excel(os.path.join(out_dir, "train.xlsx"), index=False)
pd.DataFrame(X_va, columns=ml_feature_cols).assign(label=y_va).to_excel(os.path.join(out_dir, "val.xlsx"), index=False)
pd.DataFrame(X_te, columns=ml_feature_cols).assign(label=y_te).to_excel(os.path.join(out_dir, "test.xlsx"), index=False)

# ============== DL：构造固定长度序列（Length, delta_t, direction） ==============
print("[Dataset2] Building sequences for DL (1D-CNN/LSTM) …")
X_seq_list: List[np.ndarray] = []
y_seq_list: List[int] = []
kept_flows: List[str] = []

for fid, g in df.sort_values(["flow_id","_time_s"]).groupby("flow_id"):
    if len(g) < MIN_PKTS_FOR_DL:
        continue
    arr_len = g["_len"].to_numpy(dtype=float)
    arr_dt  = g["delta_t"].to_numpy(dtype=float)
    arr_dir = g["_dir_sign"].to_numpy(dtype=float)

    # 组装 [T, C]
    seq = np.stack([arr_len, arr_dt, arr_dir], axis=1)  # [T, 3]

    # 截断/填充至 SEQ_LEN
    T = seq.shape[0]
    if T >= SEQ_LEN:
        seq_fixed = seq[:SEQ_LEN]
    else:
        pad = np.zeros((SEQ_LEN - T, seq.shape[1]), dtype=seq.dtype)
        seq_fixed = np.vstack([seq, pad])

    X_seq_list.append(seq_fixed)
    y_seq_list.append(int(flows_df.loc[flows_df["flow_id"] == fid, "y"].values[0]))
    kept_flows.append(fid)

if len(X_seq_list) > 0:
    X_seq = np.stack(X_seq_list, axis=0)  # [N, L, C]
    y_seq = np.array(y_seq_list, dtype=int)

    # 切分
    strat = y_seq if len(np.unique(y_seq)) > 1 else None
    Xtr, Xtmp, ytr, ytmp = train_test_split(
        X_seq, y_seq, test_size=0.30, random_state=42, stratify=strat
    )
    strat2 = ytr if len(np.unique(ytr)) > 1 else None
    Xva, Xte, yva, yte = train_test_split(
        Xtr, ytr, test_size=0.50, random_state=42, stratify=strat2
    )

    # 保存 DL .npz
    np.savez_compressed(
        os.path.join(out_dir, "dl_arrays_seq.npz"),
        X_train_seq=Xtr, y_train=ytr,
        X_val_seq=Xva, y_val=yva,
        X_test_seq=Xte, y_test=yte,
        channels=np.array(DL_CHANNELS),
        seq_len=np.array([SEQ_LEN])
    )
    print(f"[Dataset2] DL sequences saved: {Xtr.shape} train, {Xva.shape} val, {Xte.shape} test")
else:
    print("[Dataset2] Not enough flows for DL sequences (after MIN_PKTS_FOR_DL filter). Skipped.")

print("[Dataset2] Done.")
