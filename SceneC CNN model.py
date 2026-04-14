# -*- coding: utf-8 -*-
"""
Scenario C (Scheme B): Multi-domain learning with feature derivation to expand common fields
Datasets (Windows paths):
- E:/USA/AIRS/AIRS WEEK/WEEK10/Network_dataset_1.xlsx      -> ds1
- E:/USA/AIRS/AIRS WEEK/WEEK10/matched_columns_file_TII_B.xlsx -> ds3
- E:/USA/AIRS/AIRS WEEK/WEEK10/matched_columns_file_BCCC_B.xlsx -> ds4

Pipeline:
1) Harmonize + derive common flow features from XLSX.
2) Take intersection across three datasets; select K (2/3/5) by priority; if <K, auto-degrade to available.
3) Build k×k outer-product images with a single StandardScaler fitted on the merged table (domain-invariant scaling).
4) Merge three datasets -> stratified 70/15/15 split.
5) Train CNN (early stopping), evaluate (Acc/Prec/Rec/F1/ROC-AUC) and compute metrics (train time, latency, RSS/GPU).
6) Save ROC, confusion matrix, and two bar charts.
"""

import os, re, json, time, numpy as np, pandas as pd, psutil, matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, confusion_matrix, roc_curve)
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# =========================
#        CONFIG
# =========================
# 原始 .xlsx（方案B直接使用 XLSX，不依赖 npz 一致性）
DS1_XLSX = r"E:\USA\AIRS\AIRS WEEK\WEEK10\Network_dataset_1.xlsx"
DS3_XLSX = r"E:\USA\AIRS\AIRS WEEK\WEEK10\matched_columns_file_TII_B.xlsx"
DS4_XLSX = r"E:\USA\AIRS\AIRS WEEK\WEEK10\matched_columns_file_BCCC_B.xlsx"

# 选择字段数（2、3、5 之一；不足时会“自动降级”）
K = 5

# 字段优先级（语义归一后的规范名）
FEATURE_PRIORITY = [
    "duration","bytes_total","packet_count",
    "pkt_size_mean","pkt_size_std",
    "iat_mean","iat_std",
    "active_mean","idle_mean"
]

OUT_DIR = r"E:\USA\AIRS\AIRS WEEK\WEEK10\experiments\scenarioC_cnn_schemeB"
os.makedirs(OUT_DIR, exist_ok=True)

# 训练参数
MAX_EPOCHS = 20
BATCH_TRAIN = 64
BATCH_EVAL  = 256
LR = 1e-3
EARLY_STOP_PATIENCE = 5
SEED = 42

# =========================
#     UTILS & MODEL
# =========================
def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024**2)

def to_nchw(X: np.ndarray, y: np.ndarray, device):
    X = torch.tensor(X, dtype=torch.float32, device=device).permute(0,3,1,2)  # [N,C,H,W]
    y = torch.tensor(y, dtype=torch.long, device=device)
    return X, y

class SmallCNN(nn.Module):
    def __init__(self, in_ch=1, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(32, n_classes)
    def forward(self, x):
        x = self.net(x)               # [B,32,1,1]
        x = x.view(x.size(0), -1)     # [B,32]
        return self.fc(x)

def save_bar_charts(out_dir: str, metrics: dict):
    def _safe_num(x):
        return (np.nan, True) if x is None else (float(x), False)

    # 分类指标图
    cls_names = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    cls_vals_raw = [metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"], metrics["roc_auc"]]
    cls_vals, cls_na = zip(*[_safe_num(v) for v in cls_vals_raw])

    plt.figure(figsize=(7.2,4.6))
    bars = plt.bar(cls_names, [0 if np.isnan(v) else v for v in cls_vals])
    plt.ylim(0,1.05); plt.ylabel("Score"); plt.title("Classification Metrics")
    for b, v, na in zip(bars, cls_vals, cls_na):
        h = b.get_height()
        if na:
            plt.text(b.get_x()+b.get_width()/2, h+0.02, "N/A", ha="center", va="bottom", fontsize=9, color="gray")
            b.set_alpha(0.5)
        else:
            plt.text(b.get_x()+b.get_width()/2, h+0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "metrics_bar_classification.png"), dpi=160); plt.close()

    # 计算指标图
    comp_names = ["Train Time (s)", "Avg Inference (ms)", "Peak GPU (MB)", "RSS Δ (MB)"]
    rss_delta = metrics["rss_mb_after"] - metrics["rss_mb_before"]
    gpu_peak_mb = metrics["gpu_peak_mb"] if metrics["gpu_peak_mb"] is not None else 0.0
    comp_vals = [metrics["train_time_sec"], metrics["inference_avg_ms_per_sample"], gpu_peak_mb, rss_delta]

    plt.figure(figsize=(7.2,4.6))
    bars = plt.bar(comp_names, comp_vals)
    plt.ylabel("Value"); plt.title("Computation Metrics")
    for b, v in zip(bars, comp_vals):
        plt.text(b.get_x()+b.get_width()/2, b.get_height(), f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "metrics_bar_computation.png"), dpi=160); plt.close()

# =========================
#   Harmonize + Derivation
# =========================
CANON_SYNONYMS = [
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
LABEL_CANDIDATES = ["label","Label","type","Traffic_Type","y"]
NON_FEATURE_DROP = {
    "ts","timestamp","time","src","dst","src_ip","dst_ip","source","destination","service",
    "dns_query","ssl_subject","ssl_issuer","http_method","http_uri","http_referrer",
    "http_user_agent","http_orig_mime_types","http_resp_mime_types","weird_name","weird_addl",
    "info","no.","_time_s"
}

def _norm(n: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_ ]+","",str(n)).strip().lower().replace(" ","_")

def _canon(n: str) -> str:
    for pat, cn in CANON_SYNONYMS:
        if re.fullmatch(pat, n): return cn
    return n

def _detect_label(df: pd.DataFrame) -> str:
    for c in LABEL_CANDIDATES:
        if c in df.columns: return c
    raise ValueError("Label column not found.")

def harmonize_xlsx(path: str) -> pd.DataFrame:
    """
    读取 xlsx，语义归一 -> 数值特征 + label（列名为规范名）;
    自动派生: packet_count, bytes_total, pkt_size_mean.
    """
    raw = pd.read_excel(path)
    raw_cols = list(raw.columns)
    norm_cols = [_norm(c) for c in raw_cols]
    df = raw.copy(); df.columns = norm_cols

    lab = _detect_label(df)

    # 应用同义词到规范名
    mapped = []
    for c in df.columns:
        mapped.append("label" if c == lab else _canon(c))
    df.columns = mapped

    # ====== 自动派生，扩大交集 ======
    # packet_count
    if set(["subflow_fwd_packets","subflow_bwd_packets"]).issubset(df.columns):
        df["packet_count"] = pd.to_numeric(df["subflow_fwd_packets"], errors="coerce").fillna(0) + \
                             pd.to_numeric(df["subflow_bwd_packets"], errors="coerce").fillna(0)

    # bytes_total
    if set(["subflow_fwd_bytes","subflow_bwd_bytes"]).issubset(df.columns):
        df["bytes_total"] = pd.to_numeric(df["subflow_fwd_bytes"], errors="coerce").fillna(0) + \
                            pd.to_numeric(df["subflow_bwd_bytes"], errors="coerce").fillna(0)

    # pkt_size_mean（有了 bytes_total 和 packet_count 才能派生）
    if "bytes_total" in df.columns and "packet_count" in df.columns:
        pc = pd.to_numeric(df.get("packet_count"), errors="coerce").fillna(0)
        bt = pd.to_numeric(df.get("bytes_total"), errors="coerce").fillna(0.0)
        df["pkt_size_mean"] = (bt / pc.replace(0, np.nan)).fillna(0.0)

    # ====== 丢弃明显非特征列 ======
    drop = {_norm(x) for x in NON_FEATURE_DROP}
    feat_cols = [c for c in df.columns if c not in drop and c != "label"]

    # 仅保留数值型
    df_feat = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # 标签二值化（如需多分类自行替换）
    y = df["label"].astype(str).str.lower().map(lambda s: 0 if s in {"benign","normal","non-encrypted"} else 1)
    df_feat["label"] = y.astype(int)
    return df_feat

def build_images_from_df(df: pd.DataFrame, feats: List[str], scaler: Optional[StandardScaler]=None):
    X_tab = df[feats].astype(np.float32).values
    y = df["label"].astype(int).values
    if scaler is None:
        scaler = StandardScaler().fit(X_tab)
    X_std = scaler.transform(X_tab)
    imgs = np.einsum("ni,nj->nij", X_std, X_std)[..., None].astype(np.float32)
    return imgs, y, scaler

# =========================
#          MAIN
# =========================
def main():
    set_seed(SEED)

    # 1) Harmonize + derive
    df1 = harmonize_xlsx(DS1_XLSX)
    df3 = harmonize_xlsx(DS3_XLSX)
    df4 = harmonize_xlsx(DS4_XLSX)

    # 2) 共同字段交集（去掉 label）
    inter_feats = list(set(df1.columns) & set(df3.columns) & set(df4.columns))
    inter_feats = [c for c in inter_feats if c != "label"]
    if not inter_feats:
        raise ValueError("No common features across the three XLSX datasets after harmonization+derivation.")

    # 3) 在交集里按优先级挑选 K 个；若不足 K，自动降级
    feats_sorted = [f for f in FEATURE_PRIORITY if f in inter_feats]
    extras = [f for f in sorted(inter_feats) if f not in feats_sorted]
    feats_sorted.extend(extras)
    K_actual = min(K, len(feats_sorted))
    if K_actual == 0:
        raise ValueError("No usable common features after harmonization.")
    use_feats = feats_sorted[:K_actual]
    print(f"[INFO] Common features selected (K_actual={K_actual}): {use_feats}")

    # 4) 用三者合并的表拟合统一 StandardScaler，再各自构建图像
    Xtab_all = np.vstack([
        df1[use_feats].astype(np.float32).values,
        df3[use_feats].astype(np.float32).values,
        df4[use_feats].astype(np.float32).values
    ])
    scaler = StandardScaler().fit(Xtab_all)

    X1, y1, _ = build_images_from_df(df1, use_feats, scaler=scaler)
    X3, y3, _ = build_images_from_df(df3, use_feats, scaler=scaler)
    X4, y4, _ = build_images_from_df(df4, use_feats, scaler=scaler)

    # 5) 多域合并 -> Stratified split (70/15/15)
    X_all = np.concatenate([X1, X3, X4], axis=0)
    y_all = np.concatenate([y1, y3, y4], axis=0)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_all, test_size=0.30, random_state=SEED, stratify=y_all
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
    )

    # 6) 训练 CNN（早停）
    device = get_device()
    Xtr_t, ytr_t = to_nchw(X_train, y_train, device)
    Xva_t, yva_t = to_nchw(X_val,   y_val,   device)
    Xte_t, yte_t = to_nchw(X_test,  y_test,  device)

    C = int(Xtr_t.shape[1])
    num_classes = int(max(y_train.max(), y_val.max(), y_test.max()) + 1)
    print(f"[INFO] Channels(C)={C}, num_classes={num_classes}, total N={len(y_all)}")

    train_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=BATCH_TRAIN, shuffle=True)
    val_loader   = DataLoader(TensorDataset(Xva_t, yva_t), batch_size=BATCH_EVAL)
    test_loader  = DataLoader(TensorDataset(Xte_t, yte_t), batch_size=BATCH_EVAL)

    classes, counts = np.unique(y_train, return_counts=True)
    weights = torch.ones(int(classes.max()+1), device=device, dtype=torch.float32)
    weights[classes] = torch.tensor((counts.max()/counts), device=device, dtype=torch.float32)

    model = SmallCNN(in_ch=C, n_classes=num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = float("inf"); best_state = None; wait = 0
    t0 = time.perf_counter(); before = rss_mb()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for ep in range(MAX_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += criterion(model(xb), yb).item()
        val_loss /= max(1, len(val_loader))
        print(f"[Epoch {ep+1:02d}] val_loss={val_loss:.6f}")

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= EARLY_STOP_PATIENCE:
                print("[INFO] Early stopping.")
                break

    train_time = time.perf_counter() - t0
    after = rss_mb()
    gpu_peak = torch.cuda.max_memory_allocated()/(1024**2) if device.type == "cuda" else None

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_model.pt"))

    # 7) 测试 + 评估
    model.eval(); probs=[]; preds=[]; labels=[]
    t1 = time.perf_counter()
    with torch.no_grad():
        for xb, yb in test_loader:
            lg = model(xb)
            pr = torch.argmax(lg, dim=1).detach().cpu().numpy()
            preds.append(pr)
            labels.append(yb.detach().cpu().numpy())
            if lg.shape[1] >= 2:
                pb = torch.softmax(lg, dim=1)[:,1].detach().cpu().numpy()
            else:
                pb = np.zeros(len(pr), dtype=float)
            probs.append(pb)
    infer = time.perf_counter() - t1

    probs = np.concatenate(probs) if len(probs)>0 else np.zeros((len(y_test),), dtype=float)
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)

    # 二分类默认；若多类请改 average="macro" 或 "weighted"
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)

    roc = None
    try:
        if len(np.unique(labels)) == 2 and np.any(probs):
            roc = roc_auc_score(labels, probs)
            fpr, tpr, _ = roc_curve(labels, probs)
            plt.figure()
            plt.plot(fpr, tpr, linewidth=2)
            plt.plot([0,1],[0,1],"--")
            plt.title("ROC Curve"); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
            plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "roc_curve.png"), dpi=140); plt.close()
    except Exception:
        roc = None

    cm = confusion_matrix(labels, preds)
    np.savetxt(os.path.join(OUT_DIR, "confusion_matrix.csv"), cm, fmt="%d", delimiter=",")

    avg_ms = (infer / len(labels)) * 1000.0
    metrics = dict(
        scenario="C (Scheme B)",
        datasets=["ds1(Network_dataset_1)","ds3(TII_B)","ds4(BCCC_B)"],
        features_used=use_feats,
        K_actual=int(len(use_feats)),
        image_shape=list(X_all.shape[1:]),
        num_samples_total=int(len(y_all)),
        split={"train":int(len(y_train)),"val":int(len(y_val)),"test":int(len(y_test))},
        train_time_sec=round(train_time,4),
        inference_total_sec=round(infer,4),
        inference_avg_ms_per_sample=round(avg_ms,4),
        rss_mb_before=round(before,2), rss_mb_after=round(after,2),
        gpu_peak_mb=round(gpu_peak,2) if gpu_peak is not None else None,
        accuracy=round(acc,4), precision=round(prec,4), recall=round(rec,4), f1=round(f1,4),
        roc_auc=round(roc,4) if roc is not None else None
    )
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    # 条形图
    save_bar_charts(OUT_DIR, metrics)
    print(f"[DONE] Artifacts saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()
