import os, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

in_path = ""       # ← Replace this with your full file path.
out_dir = "./processed/dataset1_full"
os.makedirs(out_dir, exist_ok=True)

# Read the entire Excel file (if local memory is limited, you can first convert it to CSV and then process it in chunks using `pandas.read_csv(chunksize=...)`).
df = pd.read_excel(in_path)  # Full Read

# Tags: Normal=0, Other=1
df["label"] = df["type"].apply(lambda x: 0 if str(x).lower()=="normal" else 1)

non_features = [
    "ts","src_ip","dst_ip","service","dns_query","ssl_subject","ssl_issuer",
    "http_method","http_uri","http_referrer","http_user_agent",
    "http_orig_mime_types","http_resp_mime_types","weird_name","weird_addl",
    "type"
]
feature_cols = [c for c in df.columns if c not in non_features and c != "label"]

X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
y = df["label"].astype(int).to_numpy()
feature_cols = X.columns.tolist()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

np.savez_compressed(
    os.path.join(out_dir, "ml_arrays.npz"),
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    X_test=X_test, y_test=y_test,
    feature_names=np.array(feature_cols)
)

import pandas as pd
pd.DataFrame(X_train, columns=feature_cols).assign(label=y_train).to_excel(os.path.join(out_dir,"train.xlsx"), index=False)
pd.DataFrame(X_val, columns=feature_cols).assign(label=y_val).to_excel(os.path.join(out_dir,"val.xlsx"), index=False)
pd.DataFrame(X_test, columns=feature_cols).assign(label=y_test).to_excel(os.path.join(out_dir,"test.xlsx"), index=False)

with open(os.path.join(out_dir, "feature_names.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(feature_cols))

print("Done:", out_dir)


