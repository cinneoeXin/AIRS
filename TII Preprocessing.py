# -*- coding: utf-8 -*-
"""
Preprocess for Dataset 3: matched_columns_file_TII_B.xlsx (flow-level)
Outputs:
- ./processed/dataset3/ml_arrays.npz
- ./processed/dataset3/train.xlsx, val.xlsx, test.xlsx
- ./processed/dataset3/scaler.pkl
"""
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 路径与输出目录
in_path = "E:/USA/AIRS/AIRS WEEK/WEEK10/matched_columns_file_TII_B.xlsx"   # ← 改成你的实际路径
out_dir = "./processed/dataset3"
os.makedirs(out_dir, exist_ok=True)

# 读取全量
df = pd.read_excel(in_path)

# 统一列名：空格→下划线
df = df.rename(columns={c: c.strip().replace(" ", "_") for c in df.columns})

# 标签列
label_col = "Label" if "Label" in df.columns else "label"
if label_col not in df.columns:
    raise ValueError("未找到标签列 Label")

# 标签二值化（如需多分类，改这里）
def encode_label(v):
    s = str(v).strip().lower()
    return 0 if s in {"benign", "normal"} else 1

y = df[label_col].apply(encode_label).astype(int).to_numpy()

# 构造数值特征矩阵：排除明显非数值列（如 Timestamp）
non_features = {label_col, "Timestamp"}
all_cols = [c for c in df.columns if c not in non_features]
X_df = df[all_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)

feature_cols = X_df.columns.tolist()
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

# 保存 .npz（供模型直接使用）
np.savez_compressed(
    os.path.join(out_dir, "ml_arrays.npz"),
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    X_test=X_test, y_test=y_test,
    feature_names=np.array(feature_cols)
)

# 另存 .xlsx（便于人工检查）
pd.DataFrame(X_train, columns=feature_cols).assign(label=y_train).to_excel(os.path.join(out_dir, "train.xlsx"), index=False)
pd.DataFrame(X_val, columns=feature_cols).assign(label=y_val).to_excel(os.path.join(out_dir, "val.xlsx"), index=False)
pd.DataFrame(X_test, columns=feature_cols).assign(label=y_test).to_excel(os.path.join(out_dir, "test.xlsx"), index=False)

print("Done:", out_dir)
