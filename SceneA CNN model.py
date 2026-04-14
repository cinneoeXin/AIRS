#Dataset3 TII
# -*- coding: utf-8 -*-
# The path has been changed to `ds3`; otherwise, it remains largely identical to `ds1`.
import os, json, time, psutil, numpy as np, torch, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, roc_curve
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

IN_PATH = "E:/cnn_images/ds3/images.npz"   # ← Change this to your path.
OUT_DIR  = "./cnn_images/ds3/artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        x = self.net(x)          # [B, 32, 1, 1]
        x = x.view(x.size(0), -1)
        return self.fc(x)

def to_torch_NCHW(X, y, device):
    # 从 (N, H, W, C) 变为 (N, C, H, W)
    X = torch.tensor(X, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
    y = torch.tensor(y, dtype=torch.long,   device=device)
    return X, y

def rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024**2)

# ---------------- Load ----------------
data = np.load(IN_PATH)
X, y = data["X_images"], data["y"]   # X: (N, H, W, C)

# Hierarchical Division
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val,   X_test, y_val, y_test   = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

device = get_device()
Xtr, ytr = to_torch_NCHW(X_train, y_train, device)   # -> (N, C, H, W)
Xva, yva = to_torch_NCHW(X_val,   y_val,   device)
Xte, yte = to_torch_NCHW(X_test,  y_test,  device)

# >>> Automatically retrieve the number of channels C and the number of classes. num_classes
C = int(Xtr.shape[1])                    # e.g., 4
num_classes = int(ytr.max().item()) + 1  # Assume labels start from 0.

train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True)
val_loader   = DataLoader(TensorDataset(Xva, yva), batch_size=256)
test_loader  = DataLoader(TensorDataset(Xte, yte), batch_size=256)

# Class Weights (Can Mitigate Imbalance)
classes, counts = np.unique(y_train, return_counts=True)
y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
class_counts = torch.bincount(y_train_t, minlength=num_classes)
class_weights = (class_counts.max().float() / class_counts.clamp_min(1).float()).to(device)

# ---------------- Model & Optim ----------------
model = SmallCNN(in_ch=C, n_classes=num_classes).to(device)   # <<< Key Modifications：in_ch = C
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------------- Train (early stopping) ----------------
best_val, best_state, patience, wait = float("inf"), None, 5, 0
t0 = time.perf_counter()
before = rss_mb()
if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats()

for epoch in range(20):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

    # val
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            val_loss += criterion(model(xb), yb).item()
    val_loss /= max(1, len(val_loader))

    if val_loss < best_val - 1e-5:
        best_val = val_loss
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            break

train_time = time.perf_counter() - t0
after = rss_mb()
gpu_peak = torch.cuda.max_memory_allocated()/(1024**2) if device.type == "cuda" else None

# Restore to Optimal
model.load_state_dict(best_state)
torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_model.pt"))

# ---------------- Test & Metrics ----------------
# ---------- Test ----------
model.eval(); probs=[]; preds=[]; labels=[]
t1 = time.perf_counter()
with torch.no_grad():
    for xb, yb in test_loader:
        lg = model(xb)
        pr = torch.argmax(lg, dim=1).detach().cpu().numpy()
        preds.append(pr)
        labels.append(yb.detach().cpu().numpy())
        # Probabilities (for ROC-AUC, if binary classification)
        if lg.shape[1] >= 2:
            pb = torch.softmax(lg, dim=1)[:, 1].detach().cpu().numpy()
        else:
            pb = np.zeros(len(pr), dtype=float)
        probs.append(pb)
infer = time.perf_counter() - t1

probs = np.concatenate(probs) if len(probs)>0 else np.zeros((len(y_test),), dtype=float)
preds = np.concatenate(preds)
labels = np.concatenate(labels)

acc = accuracy_score(labels, preds)
prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)

roc = None
try:
    # ROC-AUC is calculated only when there are exactly two classes and the probabilities are valid.
    if len(np.unique(labels)) == 2 and np.any(probs):
        roc = roc_auc_score(labels, probs)
        fpr, tpr, _ = roc_curve(labels, probs)
        plt.figure()
        plt.plot(fpr, tpr, linewidth=2)
        plt.plot([0,1],[0,1],"--")
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "roc_curve.png"), dpi=140)
        plt.close()
except Exception:
    roc = None

cm = confusion_matrix(labels, preds)
np.savetxt(os.path.join(OUT_DIR, "A_confusion_matrix.csv"), cm, fmt="%d", delimiter=",")

avg_ms = (infer / len(labels)) * 1000.0
metrics = dict(
    dataset="ds3", num_samples=int(len(y)), image_shape=list(X.shape[1:]),
    train_time_sec=round(train_time,4), inference_total_sec=round(infer,4),
    inference_avg_ms_per_sample=round(avg_ms,4),
    rss_mb_before=round(before,2), rss_mb_after=round(after,2),
    gpu_peak_mb=round(gpu_peak,2) if gpu_peak is not None else None,
    accuracy=round(acc,4), precision=round(prec,4), recall=round(rec,4), f1=round(f1,4),
    roc_auc=round(roc,4) if roc is not None else None,
    classes={int(c): int(n) for c, n in zip(classes, counts)}
)
with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print(json.dumps(metrics, indent=2))

# ---------- NEW: Bar Chart Visualization ----------
def _safe_num(x):
    # Converts `None` to `np.nan` for plotting purposes; returns the numerical value and an N/A flag.
    return (np.nan, True) if x is None else (float(x), False)

# 1) Categorical Metric Bar Chart
cls_names = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
cls_vals_raw = [metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"], metrics["roc_auc"]]
cls_vals, cls_na = zip(*[_safe_num(v) for v in cls_vals_raw])

plt.figure(figsize=(7, 4.5))
bars = plt.bar(cls_names, [0 if np.isnan(v) else v for v in cls_vals])
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("Classification Metrics")

# Annotate values ​​on the columns; use gray text for N/A.
for b, v, na in zip(bars, cls_vals, cls_na):
    h = b.get_height()
    if na:
        plt.text(b.get_x()+b.get_width()/2, h+0.01, "N/A", ha="center", va="bottom", fontsize=9, color="gray")
        b.set_alpha(0.5)
    else:
        plt.text(b.get_x()+b.get_width()/2, h+0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "A_metrics_bar_classification.png"), dpi=160)
plt.close()

# 2) Calculated Metrics Bar Chart
comp_names = ["Train Time (s)", "Avg Inference (ms)", "Peak GPU (MB)", "RSS Δ (MB)"]
# Calculate the change in RSS (after - before); this may be negative (indicating memory release).
rss_delta = metrics["rss_mb_after"] - metrics["rss_mb_before"]
gpu_peak_mb = metrics["gpu_peak_mb"] if metrics["gpu_peak_mb"] is not None else 0.0
comp_vals = [metrics["train_time_sec"], metrics["inference_avg_ms_per_sample"], gpu_peak_mb, rss_delta]

plt.figure(figsize=(7, 4.5))
bars = plt.bar(comp_names, comp_vals)
plt.ylabel("Value")
plt.title("Computation Metrics")

# Annotate Values
for b, v in zip(bars, comp_vals):
    plt.text(b.get_x()+b.get_width()/2, b.get_height(),
             f"{v:.2f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "A_metrics_bar_computation.png"), dpi=160)
plt.close()


