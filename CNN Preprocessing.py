# Network dataset
# save as: ds1_feature_images.py
# save as: ds1_feature_images_fix.py
import os, re, numpy as np, pandas as pd, pickle
from sklearn.preprocessing import StandardScaler

IN_PATH = ""   # ← Change this to your path.
OUT_DIR = "./cnn_images/ds1"
K = 5                                # The number of desired fields：2 / 3 / 5

os.makedirs(OUT_DIR, exist_ok=True)

# —— Rules: Column Name Normalization + Synonyms
CANON_SYNONYMS = [
    (r"^(flow_)?duration$", "duration"),
    (r"^duration$", "duration"),
    (r"^(len|length|bytes)_(mean|avg|average)$", "pkt_size_mean"),
    (r"^(len|length|bytes)_std$",  "pkt_size_std"),
    (r"^(len|length|bytes)_min$",  "pkt_size_min"),
    (r"^(len|length|bytes)_max$",  "pkt_size_max"),
    (r"^(iat|inter[_ ]?arrival[_ ]?time)_(mean)$", "iat_mean"),
    (r"^(iat|inter[_ ]?arrival[_ ]?time)_(std)$",  "iat_std"),
    (r"^(iat|inter[_ ]?arrival[_ ]?time)_(min)$",  "iat_min"),
    (r"^(iat|inter[_ ]?arrival[_ ]?time)_(max)$",  "iat_max"),
    (r"^(bytes_total|total_bytes)$", "bytes_total"),
    (r"^(pkt_cnt|packet_count|packets)$", "packet_count"),
    # DS1 Common Fields (for Fallback Selection / Derivation)
    (r"^src_bytes$", "src_bytes"),
    (r"^dst_bytes$", "dst_bytes"),
    (r"^src_pkts$",  "src_pkts"),
    (r"^dst_pkts$",  "dst_pkts"),
    (r"^missed_bytes$", "missed_bytes"),
    (r"^src_ip_bytes$", "src_ip_bytes"),
    (r"^dst_ip_bytes$", "dst_ip_bytes"),
]

NON_FEATURE = {
    "ts","timestamp","time","src_ip","dst_ip","source","destination","service",
    "dns_query","ssl_subject","ssl_issuer","http_method","http_uri","http_referrer",
    "http_user_agent","http_orig_mime_types","http_resp_mime_types","weird_name","weird_addl",
    "type","label","info","no."
}
LABEL_CANDS = ["label","type","Label","Traffic_Type","y"]

def norm_name(s):
    return re.sub(r"[^a-zA-Z0-9_ ]+","",str(s)).strip().lower().replace(" ","_")

def canonize(n):
    for pat, cn in CANON_SYNONYMS:
        if re.fullmatch(pat, n): return cn
    return n

# 1) Reading Data
df = pd.read_excel(IN_PATH)

# 2) Labels → Binary（normal/benign/non-encrypted=0，other=1）
labcol = None
for c in LABEL_CANDS:
    if c in df.columns:
        labcol = c; break
if labcol is None:
    raise ValueError("Label column not found.")
y = df[labcol].astype(str).str.lower().map(lambda s: 0 if s in {"normal","benign","non-encrypted"} else 1).values

# 3) Column Name Normalization / Semantic Alignment
raw_cols = list(df.columns)
norm_cols = [norm_name(c) for c in raw_cols]
df2 = df.copy(); df2.columns = norm_cols
mapped = [("label" if c==norm_name(labcol) else canonize(c)) for c in df2.columns]
df2.columns = mapped

# 4) Automatically derive missing key fields.
# bytes_total
if "bytes_total" not in df2.columns:
    if {"src_bytes","dst_bytes"}.issubset(df2.columns):
        df2["bytes_total"] = pd.to_numeric(df2["src_bytes"], errors="coerce").fillna(0) + \
                             pd.to_numeric(df2["dst_bytes"], errors="coerce").fillna(0)
# packet_count
if "packet_count" not in df2.columns:
    if {"src_pkts","dst_pkts"}.issubset(df2.columns):
        df2["packet_count"] = pd.to_numeric(df2["src_pkts"], errors="coerce").fillna(0) + \
                              pd.to_numeric(df2["dst_pkts"], errors="coerce").fillna(0)
# pkt_size_mean
if "pkt_size_mean" not in df2.columns:
    if {"bytes_total","packet_count"}.issubset(df2.columns):
        denom = pd.to_numeric(df2["packet_count"], errors="coerce").replace(0, np.nan)
        df2["pkt_size_mean"] = pd.to_numeric(df2["bytes_total"], errors="coerce") / denom
        df2["pkt_size_mean"] = df2["pkt_size_mean"].fillna(0.0)

# 5) Field Selection: Start with "Priority Fields"; if insufficient, use "Common Alternatives"; if still insufficient, use "High-Variance Numerical Columns."
BASE_PRIORITY = [
    "duration","bytes_total","packet_count","pkt_size_mean","pkt_size_std",
    "iat_mean","iat_std","active_mean","idle_mean"
]
EXTRA_CANDIDATES = [
    # Common in DS1 and related to encryption/stream strength.
    "src_bytes","dst_bytes","src_pkts","dst_pkts",
    "missed_bytes","src_ip_bytes","dst_ip_bytes"
]

# Constructing the Candidate Pool
avail = [f for f in BASE_PRIORITY if f in df2.columns]
if len(avail) < K:
    extras = [f for f in EXTRA_CANDIDATES if f in df2.columns and f not in avail]
    avail += extras

# If the count remains insufficient, supplement the remainder from all numerical columns, sorted in descending order of variance.
if len(avail) < K:
    # Available numerical columns (excluding non-feature columns and labels)
    ban = set(NON_FEATURE) | {"label"}
    num_cols = []
    for c in df2.columns:
        if c in ban: continue
        # Attempting to convert to a numeric value
        if pd.api.types.is_numeric_dtype(df2[c]):
            num_cols.append(c)
        else:
            try:
                pd.to_numeric(df2[c])
                num_cols.append(c)
            except Exception:
                pass
    # Calculate variance
    num_df = df2[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    variances = num_df.var().sort_values(ascending=False)
    for c in list(variances.index):
        if c not in avail:
            avail.append(c)
        if len(avail) >= K:
            break

use_feats = avail[:K]
print(f"[INFO] Using features: {use_feats}")

# 6) Quantification + Standardization
X_tab = df2[use_feats].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
scaler = StandardScaler(); X_std = scaler.fit_transform(X_tab)

# 7) Generate a k×k outer product image.
N, k = X_std.shape
imgs = np.einsum("ni,nj->nij", X_std, X_std)[..., None]  # N×k×k×1

# 8) save
np.savez_compressed(os.path.join(OUT_DIR,"images.npz"),
    X_images=imgs, y=y, feature_names=np.array(use_feats), method=np.array(["outer_product"]))
with open(os.path.join(OUT_DIR,"scaler.pkl"),"wb") as f: pickle.dump(scaler, f)
with open(os.path.join(OUT_DIR,"feature_names.txt"),"w", encoding="utf-8") as f: f.write("\n".join(use_feats))

print("DS1 OK:", OUT_DIR, "shape:", imgs.shape)


#VPN dataset
# save as: ds2_feature_images_fix.py
import os, re, numpy as np, pandas as pd, pickle
from sklearn.preprocessing import StandardScaler

IN_PATH = "E:/USA/AIRS/AIRS WEEK/WEEK10/VPN_email_classified.xlsx"   # ← Change this to your path.
OUT_DIR = "./cnn_images/ds2"
K = 5
MODE = "multi"          # "outer" or "multi"
FLOW_TIMEOUT_S = 120.0  # Stream Segmentation Timeout

os.makedirs(OUT_DIR, exist_ok=True)

LABEL_CANDS = ["Traffic_Type","label","type","Label","y"]
def norm_name(s): return re.sub(r"[^a-zA-Z0-9_ ]+","",str(s)).strip().lower().replace(" ","_")

# Read Package-Level
df = pd.read_excel(IN_PATH)
# Required Columns
COL_TIME, COL_SRC, COL_DST, COL_PROTO, COL_LEN = "Time","Source","Destination","Protocol","Length"
missing = [c for c in [COL_TIME,COL_SRC,COL_DST,COL_PROTO,COL_LEN] if c not in df.columns]
if missing: raise ValueError(f"Missing columns: {missing}")

# Label → Binary
labcol = next((c for c in LABEL_CANDS if c in df.columns), None)
if labcol is None: raise ValueError("Label column not found.")
y_pkt = df[labcol].astype(str).str.lower().map(lambda s: 0 if s in {"normal","benign","non-encrypted"} else 1).astype(int)

# Undirected Edges + Sorting
def canonical_pair(a,b):
    a=str(a); b=str(b)
    return tuple(sorted([a,b]))
df["_time_s"] = pd.to_numeric(df[COL_TIME], errors="coerce").fillna(0.0)
df["_len"]    = pd.to_numeric(df[COL_LEN],  errors="coerce").fillna(0.0)
df["_key"]    = df.apply(lambda r:(canonical_pair(r[COL_SRC],r[COL_DST]), r[COL_PROTO]), axis=1)
df = df.sort_values(by=["_key","_time_s"]).reset_index(drop=True)

# Switch Flow ID (by Timeout)
flow_ids=[]; last_time={}; flow_idx={}
for _, r in df.iterrows():
    key, t = r["_key"], r["_time_s"]
    if key not in last_time:
        last_time[key]=t; flow_idx[key]=0
    elif t - last_time[key] > FLOW_TIMEOUT_S:
        flow_idx[key]+=1
        last_time[key]=t
    else:
        last_time[key]=t
    flow_ids.append(f"{key}-{flow_idx[key]}")
df["flow_id"]=flow_ids

# Δt
df["delta_t"] = df.groupby("flow_id")["_time_s"].diff().fillna(0.0)
df["_y"] = y_pkt

# Grade-Level Statistics
grp=df.groupby("flow_id")
flows=pd.DataFrame({
    "flow_id":grp.size().index,
    "packet_count": grp.size().values,
    "bytes_total":  grp["_len"].sum().values,
    "pkt_size_mean":grp["_len"].mean().values,
    "pkt_size_std": grp["_len"].std().fillna(0.0).values,
    "iat_mean":     grp["delta_t"].mean().values,
    "iat_std":      grp["delta_t"].std().fillna(0.0).values,
    "duration":     (grp["_time_s"].max() - grp["_time_s"].min()).values,
    "label":        grp["_y"].agg(lambda x:x.value_counts().index[0]).values,
})

# Field Priority + Fallback
PRIORITY = ["duration","bytes_total","packet_count","pkt_size_mean","pkt_size_std","iat_mean","iat_std"]
EXTRA    = []  # Package-level aggregations already include key fields, generally eliminating the need for additional fallback options.
avail = [f for f in PRIORITY if f in flows.columns]
if len(avail) < K:
    extras = [f for f in EXTRA if f in flows.columns and f not in avail]
    avail += extras
if len(avail) < K:
    num_df = flows[[c for c in flows.columns if c not in {"label","flow_id"}]].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    variances = num_df.var().sort_values(ascending=False)
    for c in list(variances.index):
        if c not in avail:
            avail.append(c)
        if len(avail) >= K:
            break
use_feats = avail[:K]
print(f"[DS2] Using features: {use_feats}")

# standardization
X_tab = flows[use_feats].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
y     = flows["label"].astype(int).values
scaler = StandardScaler(); X_std = scaler.fit_transform(X_tab)

# Image Generation
def build_images(X, mode="outer"):
    # X: N×K
    outer = np.einsum("ni,nj->nij", X, X)              # N×K×K
    if mode == "outer":
        return outer[..., None]
    # multi: [outer, |outer|, row_repeat, col_repeat]
    row_rep = np.tile(X[:,None,:], (1, X.shape[1], 1))
    col_rep = np.tile(X[:,:,None], (1, 1, X.shape[1]))
    imgs = np.stack([outer, np.abs(outer), row_rep, col_rep], axis=-1)
    return imgs

imgs = build_images(X_std, MODE)

# save
np.savez_compressed(os.path.join(OUT_DIR,"images.npz"),
    X_images=imgs, y=y, feature_names=np.array(use_feats), mode=np.array([MODE]))
with open(os.path.join(OUT_DIR,"scaler.pkl"),"wb") as f: pickle.dump(scaler,f)
with open(os.path.join(OUT_DIR,"feature_names.txt"),"w") as f: f.write("\n".join(use_feats))
print("DS2 OK:", OUT_DIR, "shape:", imgs.shape)


#TII dataset
# save as: ds3_feature_images.py
import os, re, numpy as np, pandas as pd, pickle
from sklearn.preprocessing import StandardScaler

IN_PATH = ""   # ← Change this to your path.
OUT_DIR = "./cnn_images/ds3"
K = 5
MODE = "multi"    # "outer" or "multi"

os.makedirs(OUT_DIR, exist_ok=True)
LABEL_CANDS = ["label","Label","type","y"]

def norm_name(s): return re.sub(r"[^a-zA-Z0-9_ ]+","",str(s)).strip().lower().replace(" ","_")
def canonize(n):
    # Common Semantic Alignment
    rules = [
        (r"^(flow_)?duration$", "duration"),
        (r"^flow[_ ]?duration$", "duration"),
        (r"^(len|length|bytes)_(mean|avg|average)$", "pkt_size_mean"),
        (r"^(len|length|bytes)_std$",  "pkt_size_std"),
        (r"^(len|length|bytes)_min$",  "pkt_size_min"),
        (r"^(len|length|bytes)_max$",  "pkt_size_max"),
        (r"^(iat|inter[_ ]?arrival[_ ]?time)_(mean)$", "iat_mean"),
        (r"^(iat|inter[_ ]?arrival[_ ]?time)_(std)$",  "iat_std"),
        (r"^(iat|inter[_ ]?arrival[_ ]?time)_(min)$",  "iat_min"),
        (r"^(iat|inter[_ ]?arrival[_ ]?time)_(max)$",  "iat_max"),
        (r"^(bytes_total|total_bytes)$", "bytes_total"),
        (r"^(pkt_cnt|packet_count|packets)$", "packet_count"),
        (r"^active_(mean)$", "active_mean"),
        (r"^active_(std)$",  "active_std"),
        (r"^active_(min)$",  "active_min"),
        (r"^active_(max)$",  "active_max"),
        (r"^idle_(mean)$",   "idle_mean"),
        (r"^idle_(std)$",    "idle_std"),
        (r"^idle_(min)$",    "idle_min"),
        (r"^idle_(max)$",    "idle_max"),
    ]
    for pat, cn in rules:
        if re.fullmatch(pat, n): return cn
    return n

# Reading data
df = pd.read_excel(IN_PATH)

# Label → Binary
lab = next((c for c in LABEL_CANDS if c in df.columns), None)
if lab is None: raise ValueError("Label not found")
y = df[lab].astype(str).str.lower().map(lambda s: 0 if s in {"benign","normal","non-encrypted"} else 1).values

# Column Name Normalization
raw_cols=list(df.columns); norm_cols=[norm_name(c) for c in raw_cols]
tmp=df.copy(); tmp.columns=norm_cols
mapped=[("label" if c==norm_name(lab) else canonize(c)) for c in tmp.columns]
tmp.columns=mapped

# Automatic Derivation
if "bytes_total" not in tmp.columns and {"subflow_fwd_bytes","subflow_bwd_bytes"}.issubset(tmp.columns):
    tmp["bytes_total"] = pd.to_numeric(tmp["subflow_fwd_bytes"], errors="coerce").fillna(0) + \
                         pd.to_numeric(tmp["subflow_bwd_bytes"], errors="coerce").fillna(0)
if "packet_count" not in tmp.columns and {"subflow_fwd_packets","subflow_bwd_packets"}.issubset(tmp.columns):
    tmp["packet_count"] = pd.to_numeric(tmp["subflow_fwd_packets"], errors="coerce").fillna(0) + \
                          pd.to_numeric(tmp["subflow_bwd_packets"], errors="coerce").fillna(0)
if "pkt_size_mean" not in tmp.columns and {"bytes_total","packet_count"}.issubset(tmp.columns):
    denom = pd.to_numeric(tmp["packet_count"], errors="coerce").replace(0, np.nan)
    tmp["pkt_size_mean"] = pd.to_numeric(tmp["bytes_total"], errors="coerce") / denom
    tmp["pkt_size_mean"] = tmp["pkt_size_mean"].fillna(0.0)

# Select fields; provide a fallback.
PRIORITY=["duration","bytes_total","packet_count","pkt_size_mean","pkt_size_std",
          "iat_mean","iat_std","active_mean","idle_mean"]
avail=[f for f in PRIORITY if f in tmp.columns]
if len(avail)<K:
    num_cols=[c for c in tmp.columns if c not in {"label"}]
    num_df = tmp[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    variances = num_df.var().sort_values(ascending=False)
    for c in list(variances.index):
        if c not in avail:
            avail.append(c)
        if len(avail)>=K: break
use_feats=avail[:K]
print(f"[DS3] Using features: {use_feats}")

# standardization
X_tab = tmp[use_feats].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
scaler = StandardScaler(); X_std = scaler.fit_transform(X_tab)

# image
def build_images(X, mode="outer"):
    outer = np.einsum("ni,nj->nij", X, X)
    if mode=="outer": return outer[...,None]
    row_rep = np.tile(X[:,None,:], (1, X.shape[1], 1))
    col_rep = np.tile(X[:,:,None], (1, 1, X.shape[1]))
    imgs = np.stack([outer, np.abs(outer), row_rep, col_rep], axis=-1)
    return imgs

imgs = build_images(X_std, MODE)

# save
np.savez_compressed(os.path.join(OUT_DIR,"images.npz"),
    X_images=imgs, y=y, feature_names=np.array(use_feats), mode=np.array([MODE]))
with open(os.path.join(OUT_DIR,"scaler.pkl"),"wb") as f: pickle.dump(scaler,f)
with open(os.path.join(OUT_DIR,"feature_names.txt"),"w") as f: f.write("\n".join(use_feats))
print("DS3 OK:", OUT_DIR, "shape:", imgs.shape)


#BCCC dataset
# save as: ds4_feature_images.py
import os, re, numpy as np, pandas as pd, pickle
from sklearn.preprocessing import StandardScaler

IN_PATH = "E:/USA/AIRS/AIRS WEEK/WEEK10/matched_columns_file_BCCC_B.xlsx"   # ← Change this to your path.
OUT_DIR = "./cnn_images/ds4"
K = 5
MODE = "multi"    # "outer" or "multi"

os.makedirs(OUT_DIR, exist_ok=True)
LABEL_CANDS=["label","Label","type","y"]

def norm_name(s): return re.sub(r"[^a-zA-Z0-9_ ]+","",str(s)).strip().lower().replace(" ","_")
def canonize(n):
    rules = [
        (r"^(flow_)?duration$", "duration"),
        (r"^duration$", "duration"),
        (r"^(len|length|bytes)_(mean|avg|average)$", "pkt_size_mean"),
        (r"^(len|length|bytes)_std$",  "pkt_size_std"),
        (r"^(len|length|bytes)_min$",  "pkt_size_min"),
        (r"^(len|length|bytes)_max$",  "pkt_size_max"),
        (r"^(iat|inter[_ ]?arrival[_ ]?time)_(mean)$", "iat_mean"),
        (r"^(iat|inter[_ ]?arrival[_ ]?time)_(std)$",  "iat_std"),
        (r"^(iat|inter[_ ]?arrival[_ ]?time)_(min)$",  "iat_min"),
        (r"^(iat|inter[_ ]?arrival[_ ]?time)_(max)$",  "iat_max"),
        (r"^(bytes_total|total_bytes)$", "bytes_total"),
        (r"^(pkt_cnt|packet_count|packets)$", "packet_count"),
        (r"^active_(mean)$", "active_mean"),
        (r"^active_(std)$",  "active_std"),
        (r"^active_(min)$",  "active_min"),
        (r"^active_(max)$",  "active_max"),
        (r"^idle_(mean)$",   "idle_mean"),
        (r"^idle_(std)$",    "idle_std"),
        (r"^idle_(min)$",    "idle_min"),
        (r"^idle_(max)$",    "idle_max"),
    ]
    for pat, cn in rules:
        if re.fullmatch(pat, n): return cn
    return n

# Reading data
df=pd.read_excel(IN_PATH)

# Label → Binary
lab = next((c for c in LABEL_CANDS if c in df.columns), None)
if lab is None: raise ValueError("Label not found")
y = df[lab].astype(str).str.lower().map(lambda s: 0 if s in {"benign","normal","non-encrypted"} else 1).values

# Column Name Normalization
raw_cols=list(df.columns); norm_cols=[norm_name(c) for c in raw_cols]
tmp=df.copy(); tmp.columns=norm_cols
mapped=[("label" if c==norm_name(lab) else canonize(c)) for c in tmp.columns]
tmp.columns=mapped

# Automatic Derivation
if "bytes_total" not in tmp.columns and {"subflow_fwd_bytes","subflow_bwd_bytes"}.issubset(tmp.columns):
    tmp["bytes_total"] = pd.to_numeric(tmp["subflow_fwd_bytes"], errors="coerce").fillna(0) + \
                         pd.to_numeric(tmp["subflow_bwd_bytes"], errors="coerce").fillna(0)
if "packet_count" not in tmp.columns and {"subflow_fwd_packets","subflow_bwd_packets"}.issubset(tmp.columns):
    tmp["packet_count"] = pd.to_numeric(tmp["subflow_fwd_packets"], errors="coerce").fillna(0) + \
                          pd.to_numeric(tmp["subflow_bwd_packets"], errors="coerce").fillna(0)
if "pkt_size_mean" not in tmp.columns and {"bytes_total","packet_count"}.issubset(tmp.columns):
    denom = pd.to_numeric(tmp["packet_count"], errors="coerce").replace(0, np.nan)
    tmp["pkt_size_mean"] = pd.to_numeric(tmp["bytes_total"], errors="coerce") / denom
    tmp["pkt_size_mean"] = tmp["pkt_size_mean"].fillna(0.0)

# Select fields; provide a fallback.
PRIORITY=["duration","bytes_total","packet_count","pkt_size_mean","pkt_size_std",
          "iat_mean","iat_std","active_mean","idle_mean"]
avail=[f for f in PRIORITY if f in tmp.columns]
if len(avail)<K:
    num_cols=[c for c in tmp.columns if c not in {"label"}]
    num_df = tmp[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    variances = num_df.var().sort_values(ascending=False)
    for c in list(variances.index):
        if c not in avail:
            avail.append(c)
        if len(avail)>=K: break
use_feats=avail[:K]
print(f"[DS4] Using features: {use_feats}")

# standardization
X_tab = tmp[use_feats].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
scaler = StandardScaler(); X_std = scaler.fit_transform(X_tab)

# image
def build_images(X, mode="outer"):
    outer = np.einsum("ni,nj->nij", X, X)
    if mode=="outer": return outer[...,None]
    row_rep = np.tile(X[:,None,:], (1, X.shape[1], 1))
    col_rep = np.tile(X[:,:,None], (1, 1, X.shape[1]))
    imgs = np.stack([outer, np.abs(outer), row_rep, col_rep], axis=-1)
    return imgs

imgs = build_images(X_std, MODE)

# save
np.savez_compressed(os.path.join(OUT_DIR,"images.npz"),
    X_images=imgs, y=y, feature_names=np.array(use_feats), mode=np.array([MODE]))
with open(os.path.join(OUT_DIR,"scaler.pkl"),"wb") as f: pickle.dump(scaler,f)
with open(os.path.join(OUT_DIR,"feature_names.txt"),"w") as f: f.write("\n".join(use_feats))
print("DS4 OK:", OUT_DIR, "shape:", imgs.shape)
