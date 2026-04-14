# -*- coding: utf-8 -*-
"""
Preprocess for Dataset 4: matched_columns_file_BCCC_B.xlsx (flow-level)
Outputs:
- ./processed/dataset4/ml_arrays.npz
- ./processed/dataset4/train.xlsx, val.xlsx, test.xlsx
- ./processed/dataset4/scaler.pkl
"""
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 路径与输出目录
in_path = "E:/USA/AIRS/AIRS WEEK/WEEK10/matched_columns_file_BCCC_B.xlsx"   # ← 改成你的实际路径
out_dir = "./processed/dataset4"
os.makedirs(out_dir, exist_ok=True)

# 读取全量
df = pd.read_excel(in_path)

# 标签编码：Non-Encrypted=0, Encrypted=1, 其他=2（如有其他标签）
def encode_label(v):
    s = str(v).strip().lower()
    if s in {"non-encrypted"}:
        return 0
    elif s in {"encrypted"}:
        return 1
    return 2

y = df["label"].apply(encode_label).astype(int).to_numpy()

# 构造数值特征矩阵：排除非数值列
non_features = {"label", "timestamp"}
feature_cols = [c for c in df.columns if c not in non_features]
X_df = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)

X = X_df.to_numpy(dtype=float)

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# 划分（Stratified）
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# 保存 .npz
np.savez_compressed(
    os.path.join(out_dir, "ml_arrays.npz"),
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    X_test=X_test, y_test=y_test,
    feature_names=np.array(feature_cols)
)

# 保存 .xlsx（便于人工查看）
pd.DataFrame(X_train, columns=feature_cols).assign(label=y_train).to_excel(os.path.join(out_dir, "train.xlsx"), index=False)
pd.DataFrame(X_val, columns=feature_cols).assign(label=y_val).to_excel(os.path.join(out_dir, "val.xlsx"), index=False)
pd.DataFrame(X_test, columns=feature_cols).assign(label=y_test).to_excel(os.path.join(out_dir, "test.xlsx"), index=False)

print("Done:", out_dir)
