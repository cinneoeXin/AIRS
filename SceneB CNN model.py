# -*- coding: utf-8 -*-
"""
Scenario B (Cross-domain CNN) — Robust label handling
- Train:    ds3 (TII_B_harmonized)
- Validate: ds4 (BCCC_B_harmonized)
- Test:     ds1 (Network_dataset_1_harmonized)

Fixes:
1) Strong feature derivation (bytes_total/packet_count/duration/... many synonyms)
2) Global feature template selection across all three datasets with fallback K
3) Robust label handling: detect label column; map strings to ints.
   - FORCE_BINARY=True: map benign-like to 0; others to 1
   - FORCE_BINARY=False: multi-class LabelEncoder (ROC-AUC disabled)

Outputs:
- best_model.pt
- metrics.json
- confusion_matrix.csv
- roc_curve.png (if binary & computable)
- metrics_bar_classification.png / metrics_bar_computation.png
"""

import os, json, time, numpy as np, pandas as pd, psutil, torch, matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, confusion_matrix, roc_curve)
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# =========================
#        CONFIG
# =========================
# 已生成的 .npz（优先用；若与模板不一致则回退到 parquet 重建）
DS1_NPZ = r"E:/USA/AIRS/AIRS WEEK/WEEK10/cnn_images/ds1/images.npz"
DS3_NPZ = r"E:/USA/AIRS/AIRS WEEK/WEEK10/cnn_images/ds3/images.npz"
DS4_NPZ = r"E:/USA/AIRS/AIRS WEEK/WEEK10/cnn_images/ds4/images.npz"

# harmonized parquet（用于模板选择与必要时重建）
DS1_PARQUET = r"E:/USA/AIRS/AIRS WEEK/WEEK10/harmonized/scenario_b/Network_dataset_1_harmonized.parquet"
DS3_PARQUET = r"E:/USA/AIRS/AIRS WEEK/WEEK10/harmonized/scenario_b/matched_columns_file_TII_B_harmonized.parquet"
DS4_PARQUET = r"E:/USA/AIRS/AIRS WEEK/WEEK10/harmonized/scenario_b/matched_columns_file_BCCC_B_harmonized.parquet"

# 希望的字段数（不足则自动从 5 降到 1）
K_TARGET = 5

# 优先级与回退字段
FEATURE_PRIORITY = [
    "duration","bytes_total","packet_count",
    "pkt_size_mean","pkt_size_std",
    "iat_mean","iat_std",
    "active_mean","idle_mean"
]
SAFE_BACKUPS = ["src_port", "dst_port", "protocol"]

# 标签设置
FORCE_BINARY = True   # True: 把 Benign/Normal/Background 归为 0，其余为 1；False：多类编码

# 训练参数与输出
MAX_EPOCHS = 20
BATCH_TRAIN = 64
BATCH_EVAL  = 256
LR = 1e-3
EARLY_STOP_PATIENCE = 5
SEED = 42

OUT_DIR = r"E:/USA/AIRS/AIRS WEEK/WEEK10/experiments/scenarioB_cnn_fixed_labels"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
#     HELPERS
# =========================
def set_seed(seed: int = 42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024**2)

def to_nchw(X: np.ndarray, y: np.ndarray, device):
    Xt = torch.tensor(X, dtype=torch.float32, device=device).permute(0,3,1,2)
    yt = torch.tensor(y, dtype=torch.long, device=device)
    return Xt, yt

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
        x = self.net(x); x = x.view(x.size(0), -1); return self.fc(x)

def save_bar_charts(out_dir: str, metrics: dict):
    def _safe_num(x): return (np.nan, True) if x is None else (float(x), False)
    # 分类
    cls_names = ["Accuracy","Precision","Recall","F1","ROC-AUC"]
    cls_vals_raw = [metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"], metrics["roc_auc"]]
    cls_vals, cls_na = zip(*[_safe_num(v) for v in cls_vals_raw])
    plt.figure(figsize=(7.2,4.6))
    bars = plt.bar(cls_names, [0 if np.isnan(v) else v for v in cls_vals])
    plt.ylim(0,1.05); plt.ylabel("Score"); plt.title("Classification Metrics")
    for b,v,na in zip(bars,cls_vals,cls_na):
        h=b.get_height()
        if na:
            plt.text(b.get_x()+b.get_width()/2, h+0.02, "N/A", ha="center", va="bottom", fontsize=9, color="gray")
            b.set_alpha(0.5)
        else:
            plt.text(b.get_x()+b.get_width()/2, h+0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,"B_metrics_bar_classification.png"), dpi=160); plt.close()
    # 计算
    comp_names = ["Train Time (s)", "Avg Inference (ms)", "Peak GPU (MB)", "RSS Δ (MB)"]
    rss_delta = metrics["rss_mb_after"] - metrics["rss_mb_before"]
    gpu_peak_mb = metrics["gpu_peak_mb"] if metrics["gpu_peak_mb"] is not None else 0.0
    comp_vals = [metrics["train_time_sec"], metrics["inference_avg_ms_per_sample"], gpu_peak_mb, rss_delta]
    plt.figure(figsize=(7.2,4.6))
    bars = plt.bar(comp_names, comp_vals)
    plt.ylabel("Value"); plt.title("Computation Metrics")
    for b,v in zip(bars,comp_vals):
        plt.text(b.get_x()+b.get_width()/2, b.get_height(), f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,"B_metrics_bar_computation.png"), dpi=160); plt.close()

# =========================
#  Robust feature engineering
# =========================
def _num(x): return pd.to_numeric(x, errors="coerce")

def derive_standard_features(df: pd.DataFrame) -> pd.DataFrame:
    """派生统一特征；数值化 protocol/src_port/dst_port。"""
    d = df.copy()

    # ---- bytes_total ----
    if "bytes_total" not in d.columns:
        cand_pairs = [
            ("subflow_fwd_bytes","subflow_bwd_bytes"),
            ("tot_fwd_bytes","tot_bwd_bytes"),
            ("total_fwd_bytes","total_bwd_bytes"),
            ("fwd_bytes","bwd_bytes"),
            ("forward_bytes","backward_bytes"),
            ("orig_bytes","resp_bytes"),
            ("fbytes","bbytes"),
        ]
        for a,b in cand_pairs:
            if {a,b}.issubset(d.columns):
                d["bytes_total"] = _num(d[a]).fillna(0) + _num(d[b]).fillna(0)
                break
        if "bytes_total" not in d.columns:
            for single in ["total_bytes","bytes","byte_count","all_bytes"]:
                if single in d.columns:
                    d["bytes_total"] = _num(d[single]).fillna(0); break

    # ---- packet_count ----
    if "packet_count" not in d.columns:
        cand_pairs = [
            ("subflow_fwd_packets","subflow_bwd_packets"),
            ("tot_fwd_pkts","tot_bwd_pkts"),
            ("total_fwd_packets","total_bwd_packets"),
            ("fwd_pkts","bwd_pkts"),
            ("forward_pkts","backward_pkts"),
            ("fpackets","bpackets"),
        ]
        for a,b in cand_pairs:
            if {a,b}.issubset(d.columns):
                d["packet_count"] = _num(d[a]).fillna(0) + _num(d[b]).fillna(0)
                break
        if "packet_count" not in d.columns:
            for single in ["packets","pkt_cnt","packet_total","total_packets"]:
                if single in d.columns:
                    d["packet_count"] = _num(d[single]).fillna(0); break

    # ---- pkt_size_mean/std ----
    if "pkt_size_mean" not in d.columns:
        for c in ["len_mean","length_mean","bytes_mean","len_avg","length_avg","bytes_avg","pkt_size_avg"]:
            if c in d.columns:
                d["pkt_size_mean"] = _num(d[c]).fillna(0.0); break
    if "pkt_size_std" not in d.columns:
        for c in ["len_std","length_std","bytes_std"]:
            if c in d.columns:
                d["pkt_size_std"] = _num(d[c]).fillna(0.0); break

    # ---- iat_mean/std ----
    if "iat_mean" not in d.columns:
        for c in ["iat_mean","inter_arrival_time_mean","flow_iat_mean"]:
            if c in d.columns:
                d["iat_mean"] = _num(d[c]).fillna(0.0); break
    if "iat_std" not in d.columns:
        for c in ["iat_std","inter_arrival_time_std","flow_iat_std"]:
            if c in d.columns:
                d["iat_std"] = _num(d[c]).fillna(0.0); break

    # ---- duration ----
    if "duration" not in d.columns:
        for c in ["flow_duration","duration_ms","flow_duration_ms","duration"]:
            if c in d.columns:
                d["duration"] = _num(d[c]).fillna(0.0); break

    # ---- active/idle mean (optional) ----
    if "active_mean" not in d.columns:
        for c in ["active_mean","flow_active_mean"]:
            if c in d.columns:
                d["active_mean"] = _num(d[c]).fillna(0.0); break
    if "idle_mean" not in d.columns:
        for c in ["idle_mean","flow_idle_mean"]:
            if c in d.columns:
                d["idle_mean"] = _num(d[c]).fillna(0.0); break

    # ---- numeric protocol / ports ----
    if "protocol" in d.columns and not np.issubdtype(d["protocol"].dtype, np.number):
        d["protocol"] = d["protocol"].astype("category").cat.codes.astype("int32")
    for c in ["src_port","dst_port"]:
        if c in d.columns and not np.issubdtype(d[c].dtype, np.number):
            d[c] = _num(d[c]).fillna(0).astype("int32")

    return d

# =========================
#   Label handling (fix)
# =========================
LABEL_CANDIDATES = [
    "label","Label","labels","Labels","class","Class","category","Category",
    "attack_cat","attack_category","Attack","Target","target","y"
]

BENIGN_LIKE = {"benign","normal","background","clean","non-malicious","nonmalicious","legit","legitimate","good"}

def find_label_column(df: pd.DataFrame) -> Optional[str]:
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            return c
    return None

def encode_labels(series: pd.Series, force_binary: bool=True) -> Tuple[np.ndarray, dict, bool]:
    """
    将标签列编码为整数。
    - force_binary=True: benign-like -> 0, others (non-empty) -> 1
    - force_binary=False: multi-class LabelEncoder (0..K-1)
    返回：y(np.int64), mapping(dict: original->encoded), is_binary(bool)
    """
    # 若本身是数值，直接返回
    if np.issubdtype(series.dtype, np.number):
        y = series.to_numpy().astype(np.int64)
        unique = np.unique(y)
        is_bin = (len(unique) == 2)
        mapping = {}  # 数值不需要映射
        return y, mapping, is_bin

    s = series.astype(str).fillna("").str.strip()
    if force_binary:
        lower = s.str.lower()
        y = np.where(lower.isin(BENIGN_LIKE) | (lower == "0"), 0, 1).astype(np.int64)
        mapping = {"benign_like->0": list(sorted(BENIGN_LIKE))}
        return y, mapping, True
    else:
        le = LabelEncoder()
        y = le.fit_transform(s).astype(np.int64)
        mapping = {cls: int(idx) for idx, cls in enumerate(le.classes_)}
        is_bin = (len(le.classes_) == 2)
        return y, mapping, is_bin

# ---------- 全局模板选择 ----------
def choose_global_template(train_df: pd.DataFrame,
                           val_df: pd.DataFrame,
                           test_df: pd.DataFrame,
                           K_target: int,
                           priority: List[str],
                           backups: List[str]) -> Tuple[List[str], int, List[str]]:
    notes=[]
    dfs = [derive_standard_features(train_df),
           derive_standard_features(val_df),
           derive_standard_features(test_df)]

    cand = [f for f in priority if f in dfs[0].columns]
    if len(cand) < K_target:
        for b in backups:
            if b in dfs[0].columns and b not in cand:
                cand.append(b)
    if len(cand) == 0:
        raise ValueError("训练集没有可用字段（优先级+回退均不可用）。")

    for k in [K_target, 5, 4, 3, 2, 1]:
        k = min(k, len(cand))
        if k <= 0: continue
        base = cand[:k]
        pool = set(base) | set(backups)
        inter = None
        for df in dfs:
            avail = set(df.columns) & pool
            inter = avail if inter is None else (inter & avail)
        if len(inter) == 0:
            notes.append(f"k={k} 时三方公共可用集合为空")
            continue

        final=[]
        for f in base:
            if f in inter and f not in final: final.append(f)
        for b in backups:
            if len(final) >= k: break
            if b in inter and b not in final: final.append(b)
        if len(final) >= 1:
            return final[:k], k, notes

    raise ValueError("无法在三份数据中达到至少 1 个共同可用字段。")

# ---------- 构图（用模板 feats） ----------
def build_outer_images(df: pd.DataFrame, feats: List[str], scaler: Optional[StandardScaler]=None, force_binary: bool=True):
    df = derive_standard_features(df)

    # 找 label 列并编码
    label_col = find_label_column(df)
    if label_col is None:
        raise ValueError("未找到标签列（尝试的候选: {}）".format(", ".join(LABEL_CANDIDATES)))
    y, label_map, is_binary = encode_labels(df[label_col], force_binary=force_binary)

    # 检查特征并构图
    for f in feats:
        if f not in df.columns:
            raise ValueError(f"字段 {f} 缺失，无法构图（请检查派生/模板选择）。")
    X_tab = df[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    if scaler is None:
        scaler = StandardScaler().fit(X_tab)
    X_std = scaler.transform(X_tab).astype(np.float32)
    X_img = np.einsum("ni,nj->nij", X_std, X_std)[..., None]  # [N,k,k,1]

    return X_img, y, scaler, label_col, label_map, is_binary

def load_npz(npz_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    d = np.load(npz_path)
    X = d["X_images"]; y = d["y"]; feats = d["feature_names"].tolist()
    return X, y, feats

# =========================
#           MAIN
# =========================
def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    # 读 parquet 用于模板选择与（必要时）重建
    if not (os.path.exists(DS3_PARQUET) and os.path.exists(DS4_PARQUET) and os.path.exists(DS1_PARQUET)):
        raise FileNotFoundError("请确认三份 harmonized parquet 均存在。")

    df_tr_raw = pd.read_parquet(DS3_PARQUET)
    df_va_raw = pd.read_parquet(DS4_PARQUET)
    df_te_raw = pd.read_parquet(DS1_PARQUET)

    # 全局模板选择（一次看三份数据）
    feats_template, k_final, notes = choose_global_template(
        df_tr_raw, df_va_raw, df_te_raw,
        K_target=K_TARGET, priority=FEATURE_PRIORITY, backups=SAFE_BACKUPS
    )
    if k_final < K_TARGET:
        print(f"[WARN] 字段数从 {K_TARGET} 降到 {k_final}；notes={notes}")
    print("[TEMPLATE]", feats_template)

    scaler = None

    # ---- Train ----
    if os.path.exists(DS3_NPZ):
        Xtr_npz, ytr_npz, feats_npz = load_npz(DS3_NPZ)
        # 即使用 npz，也需要知道标签二/多类；因此从 parquet 取标签重新编码（保证一致性）
        Xtr, ytr, scaler, label_col_tr, label_map_tr, is_bin_tr = build_outer_images(
            df_tr_raw, feats_template, scaler=None, force_binary=FORCE_BINARY
        )
        if feats_npz == feats_template:
            # 用 npz 的 X，以 parquet 的 y（编码后）
            if Xtr_npz.shape[0] == Xtr.shape[0]:
                Xtr = Xtr_npz
            else:
                print("[WARN] ds3 npz 与 parquet 行数不同，改用 parquet 构图的 X。")
        print(f"[train] X:{Xtr.shape}, y:{ytr.shape}, label_col:{label_col_tr}, binary:{is_bin_tr}, map:{label_map_tr}")
    else:
        Xtr, ytr, scaler, label_col_tr, label_map_tr, is_bin_tr = build_outer_images(
            df_tr_raw, feats_template, scaler=None, force_binary=FORCE_BINARY
        )
        print(f"[train-build] X:{Xtr.shape}, y:{ytr.shape}, label_col:{label_col_tr}, binary:{is_bin_tr}, map:{label_map_tr}")

    # ---- Val ----
    if os.path.exists(DS4_NPZ):
        Xva_npz, yva_npz, feats_npz = load_npz(DS4_NPZ)
        Xva, yva, _, label_col_va, label_map_va, is_bin_va = build_outer_images(
            df_va_raw, feats_template, scaler=scaler, force_binary=FORCE_BINARY
        )
        if feats_npz == feats_template and Xva_npz.shape[0] == Xva.shape[0]:
            Xva = Xva_npz  # 使用 npz 的 X，但 y 统一用 parquet 编码后的
        print(f"[val]   X:{Xva.shape}, y:{yva.shape}, label_col:{label_col_va}, binary:{is_bin_va}, map:{label_map_va}")
    else:
        Xva, yva, _, label_col_va, label_map_va, is_bin_va = build_outer_images(
            df_va_raw, feats_template, scaler=scaler, force_binary=FORCE_BINARY
        )
        print(f"[val-build] X:{Xva.shape}, y:{yva.shape}, label_col:{label_col_va}, binary:{is_bin_va}, map:{label_map_va}")

    # ---- Test ----
    if os.path.exists(DS1_NPZ):
        Xte_npz, yte_npz, feats_npz = load_npz(DS1_NPZ)
        Xte, yte, _, label_col_te, label_map_te, is_bin_te = build_outer_images(
            df_te_raw, feats_template, scaler=scaler, force_binary=FORCE_BINARY
        )
        if feats_npz == feats_template and Xte_npz.shape[0] == Xte.shape[0]:
            Xte = Xte_npz
        print(f"[test]  X:{Xte.shape}, y:{yte.shape}, label_col:{label_col_te}, binary:{is_bin_te}, map:{label_map_te}")
    else:
        Xte, yte, _, label_col_te, label_map_te, is_bin_te = build_outer_images(
            df_te_raw, feats_template, scaler=scaler, force_binary=FORCE_BINARY
        )
        print(f"[test-build] X:{Xte.shape}, y:{yte.shape}, label_col:{label_col_te}, binary:{is_bin_te}, map:{label_map_te}")

    # ---- Torch ----
    device = get_device()
    Xtr_t, ytr_t = to_nchw(Xtr, ytr, device)
    Xva_t, yva_t = to_nchw(Xva, yva, device)
    Xte_t, yte_t = to_nchw(Xte, yte, device)

    C = int(Xtr_t.shape[1])
    num_classes = int(max(ytr.max(), yva.max(), yte.max()) + 1)
    print(f"[info] Channels={C}, num_classes={num_classes}, feats={feats_template}")

    train_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=BATCH_TRAIN, shuffle=True)
    val_loader   = DataLoader(TensorDataset(Xva_t, yva_t), batch_size=BATCH_EVAL)
    test_loader  = DataLoader(TensorDataset(Xte_t, yte_t), batch_size=BATCH_EVAL)

    # 类别权重（训练集）
    classes, counts = np.unique(ytr, return_counts=True)
    weights = torch.ones(int(classes.max()+1), device=device, dtype=torch.float32)
    weights[classes] = torch.tensor((counts.max()/counts), device=device, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights)

    model = SmallCNN(in_ch=C, n_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ---- Train (early stopping on val) ----
    best_val = float("inf"); best_state=None; wait=0
    t0 = time.perf_counter(); before = rss_mb()
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats()

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

    # 恢复最佳并保存
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_model.pt"))

    # ---- Test ----
    model.eval(); probs=[]; preds=[]; labels=[]
    t1 = time.perf_counter()
    with torch.no_grad():
        for xb, yb in test_loader:
            lg = model(xb)
            pr = torch.argmax(lg, dim=1).detach().cpu().numpy()
            preds.append(pr); labels.append(yb.detach().cpu().numpy())
            if lg.shape[1] >= 2:
                pb = torch.softmax(lg, dim=1)[:, 1].detach().cpu().numpy()
            else:
                pb = np.zeros(len(pr), dtype=float)
            probs.append(pb)
    infer = time.perf_counter() - t1

    probs  = np.concatenate(probs) if len(probs)>0 else np.zeros((len(yte),), dtype=float)
    preds  = np.concatenate(preds)
    labels = np.concatenate(labels)

    # 指标（默认二分类；若你设置 FORCE_BINARY=False 则这里建议改 average="macro"）
    average_mode = "binary" if FORCE_BINARY else "macro"
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average=average_mode, zero_division=0)

    roc = None
    try:
        if FORCE_BINARY and len(np.unique(labels)) == 2 and np.any(probs):
            roc = roc_auc_score(labels, probs)
            fpr, tpr, _ = roc_curve(labels, probs)
            plt.figure(); plt.plot(fpr, tpr, linewidth=2); plt.plot([0,1],[0,1],"--")
            plt.title("ROC Curve"); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
            plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "B_roc_curve.png"), dpi=140); plt.close()
    except Exception:
        roc = None

    cm = confusion_matrix(labels, preds)
    np.savetxt(os.path.join(OUT_DIR, "B_confusion_matrix.csv"), cm, fmt="%d", delimiter=",")

    avg_ms = (infer / len(labels)) * 1000.0
    metrics = dict(
        scenario="B",
        train_dataset="TII_B (ds3)",
        val_dataset="BCCC_B (ds4)",
        test_dataset="Network_dataset_1 (ds1)",
        features_used=feats_template,
        num_classes=int(num_classes),
        force_binary=FORCE_BINARY,
        image_shape=list(Xtr.shape[1:]),
        train_time_sec=round(train_time,4),
        inference_total_sec=round(infer,4),
        inference_avg_ms_per_sample=round(avg_ms,4),
        rss_mb_before=round(before,2), rss_mb_after=round(after,2),
        gpu_peak_mb=round(gpu_peak,2) if gpu_peak is not None else None,
        accuracy=round(acc,4), precision=round(prec,4), recall=round(rec,4), f1=round(f1,4),
        roc_auc=round(roc,4) if roc is not None else None,
        train_class_counts={int(c): int(n) for c, n in zip(classes, counts)}
    )
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    # 条形图
    save_bar_charts(OUT_DIR, metrics)
    print(f"[DONE] Artifacts saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()
