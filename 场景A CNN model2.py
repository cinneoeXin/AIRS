#Dataset1 Network
# -*- coding: utf-8 -*-
# save as: train_cnn_ds1.py
import os, time, random, numpy as np
from dataclasses import dataclass
from typing import Tuple
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

# ========= 路径与超参 =========
NPZ_PATH = "E:/USA/AIRS/AIRS WEEK/WEEK10/cnn_images/ds1/images.npz"   # ← 若路径不同，改这里
OUT_DIR  = "./cnn_images/ds1/artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4

# ========= 随机种子 & 设备 =========
def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========= 数据集 =========
class NpzImageDataset(Dataset):
    def __init__(self, X, y):
        # X: N×k×k×1
        self.X = torch.from_numpy(X).permute(0,3,1,2).float()  # N×1×k×k
        self.y = torch.from_numpy(y).long()
    def __len__(self): return self.X.size(0)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ========= 模型 =========
class SmallCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(64),
            nn.AdaptiveAvgPool2d(1)  # -> (N,64,1,1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ========= 载入数据 =========
data = np.load(NPZ_PATH, allow_pickle=True)
X = data["X_images"]     # N×k×k×1
y = data["y"]

num_classes = int(len(np.unique(y)))
dataset = NpzImageDataset(X, y)

# Stratified 划分 70/15/15
idx = np.arange(len(dataset))
idx_train, idx_temp, y_train, y_temp = train_test_split(idx, y, test_size=0.30, random_state=SEED, stratify=y)
idx_val, idx_test,  y_val,  y_test  = train_test_split(idx_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp)

ds_train = Subset(dataset, idx_train)
ds_val   = Subset(dataset, idx_val)
ds_test  = Subset(dataset, idx_test)

train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
val_loader   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
test_loader  = DataLoader(ds_test,  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# 类权重（对不平衡有帮助）
class_counts = np.bincount(y_train, minlength=num_classes)
class_weights = (class_counts.sum() / (class_counts + 1e-12))
class_weights = class_weights / class_weights.mean()
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

# ========= 训练要素 =========
model = SmallCNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor if num_classes>1 else None)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
best_val_f1 = -1.0
best_path = os.path.join(OUT_DIR, "best_model.pt")

def evaluate(loader) -> Tuple[float,float,float,float,float]:
    model.eval()
    all_y, all_p = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            all_y.append(yb.cpu().numpy())
            all_p.append(preds.cpu().numpy())
    y_true = np.concatenate(all_y); y_pred = np.concatenate(all_p)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    try:
        if num_classes==2:
            # 为 AUC，取正类概率
            model.eval()
            all_prob = []
            with torch.no_grad():
                for xb, yb in loader:
                    xb = xb.to(device)
                    prob = torch.softmax(model(xb), dim=1)[:,1].cpu().numpy()
                    all_prob.append(prob)
            y_score = np.concatenate(all_prob)
            auc = roc_auc_score(y_true, y_score)
        else:
            auc = float("nan")
    except Exception:
        auc = float("nan")
    return acc, prec, rec, f1, auc

# ========= 训练循环 =========
for epoch in range(1, EPOCHS+1):
    model.train()
    running = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward(); optimizer.step()
        running += loss.item() * xb.size(0)
    train_loss = running / len(ds_train)
    acc, prec, rec, f1, auc = evaluate(val_loader)
    print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val: acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f} auc={auc:.4f}")
    if f1 > best_val_f1:
        best_val_f1 = f1
        torch.save({"model":model.state_dict(),"num_classes":num_classes}, best_path)

# ========= 测试集评估 =========
ckpt = torch.load(best_path, map_location=device)
model.load_state_dict(ckpt["model"])

test_acc, test_prec, test_rec, test_f1, test_auc = evaluate(test_loader)

# 推理时延（每样本毫秒）
model.eval()
start = time.time()
n = 0
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        _ = model(xb)
        n += xb.size(0)
elapsed = time.time() - start
latency_ms = (elapsed / max(n,1)) * 1000.0

# ========= 输出 =========
with open(os.path.join(OUT_DIR,"metrics.txt"), "w") as f:
    f.write(f"test_acc={test_acc:.6f}\n")
    f.write(f"test_prec={test_prec:.6f}\n")
    f.write(f"test_rec={test_rec:.6f}\n")
    f.write(f"test_f1={test_f1:.6f}\n")
    f.write(f"test_auc={test_auc:.6f}\n")
    f.write(f"avg_inference_latency_ms_per_sample={latency_ms:.6f}\n")

print("Saved best model to:", best_path)
print(f"[TEST] acc={test_acc:.4f} prec={test_prec:.4f} rec={test_rec:.4f} f1={test_f1:.4f} auc={test_auc:.4f} latency(ms)={latency_ms:.2f}")


#Dataset2 VPN
# -*- coding: utf-8 -*-
# 同 DS1，只改路径与 dataset 名称
import os, json, time, psutil, numpy as np, torch, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, roc_curve
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

IN_PATH = "E:/USA/AIRS/AIRS WEEK/WEEK10/cnn_images/ds2/images.npz"
OUT_DIR  = "./cnn_images/ds2/artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------ Utils ------------------
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024**2)

def ensure_nchw(X):
    """
    将任意常见格式的图像张量变为 NCHW，并返回 (X_nchw, C, H, W)
    支持:
      - (N, H, W)           -> (N, 1, H, W)
      - (N, H, W, C)       -> (N, C, H, W)
      - (N, C, H, W)       -> 原样
    """
    if X.ndim == 3:
        N, H, W = X.shape
        X = X.reshape(N, 1, H, W)
    elif X.ndim == 4:
        # 判断通道位置：如果最后一维很小（<=8），一般是 NHWC
        N, a, b, c = X.shape
        # 可能是 NHWC
        if c <= 8 and a == b:
            X = np.transpose(X, (0, 3, 1, 2))  # NHWC -> NCHW
        # 如果是 NCHW（第二维较小，后两维方阵），保持不动
        # 如果写入时使用了单通道但没留最后一维，也会被上面的 ndim==3 分支处理
    else:
        raise ValueError(f"Unsupported image array shape: {X.shape}")
    return X, X.shape[1], X.shape[2], X.shape[3]  # N, C, H, W

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
        x = self.net(x)             # [B, 32, 1, 1]
        x = x.view(x.size(0), -1)   # [B, 32]
        return self.fc(x)

def to_torch(X_nchw, y, device):
    X = torch.tensor(X_nchw, dtype=torch.float32, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)
    return X, y

# ------------------ Load ------------------
data = np.load(IN_PATH)
X_raw = data["X_images"]  # 可能是 (N,H,W) / (N,H,W,C) / (N,C,H,W)
y     = data["y"]

# 转成 NCHW，并拿到真实通道数
X_nchw, C, H, W = ensure_nchw(X_raw)

# 划分
X_train, X_temp, y_train, y_temp = train_test_split(X_nchw, y, test_size=0.30, random_state=42, stratify=y)
X_val,   X_test, y_val, y_test   = train_test_split(X_temp,  y_temp, test_size=0.50, random_state=42, stratify=y_temp)

device = get_device()
Xtr, ytr = to_torch(X_train, y_train, device)
Xva, yva = to_torch(X_val,   y_val,   device)
Xte, yte = to_torch(X_test,  y_test,  device)

train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True)
val_loader   = DataLoader(TensorDataset(Xva, yva), batch_size=256)
test_loader  = DataLoader(TensorDataset(Xte, yte), batch_size=256)

# 类别权重（缓解不平衡）
classes, counts = np.unique(y_train, return_counts=True)
n_classes = int(classes.max() + 1)
weights = torch.ones(n_classes, device=device)
weights[classes] = torch.tensor((counts.max()/counts), dtype=torch.float32, device=device)
criterion = nn.CrossEntropyLoss(weight=weights)

# 模型的 in_ch 自动等于真实通道数 C
model = SmallCNN(in_ch=C, n_classes=n_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ------------------ Train (Early Stopping) ------------------
best_val, best_state, patience, wait = float("inf"), None, 5, 0
t0 = time.perf_counter()
rss_before = rss_mb()
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

    # 验证
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
rss_after = rss_mb()
gpu_peak_mb = torch.cuda.max_memory_allocated()/(1024**2) if device.type == "cuda" else None

# 恢复最佳并保存
model.load_state_dict(best_state)
os.makedirs(OUT_DIR, exist_ok=True)
torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_model.pt"))

# ------------------ Test & Metrics ------------------
model.eval()
probs, preds, labels = [], [], []
t1 = time.perf_counter()
with torch.no_grad():
    for xb, yb in test_loader:
        lg = model(xb)
        pr = torch.argmax(lg, dim=1).detach().cpu().numpy()
        # 对二分类，取第1类的概率；多分类时下行可按需改为 softmax 全量输出
        pb = torch.softmax(lg, dim=1)[:, min(1, n_classes-1)].detach().cpu().numpy()
        probs.append(pb); preds.append(pr); labels.append(yb.detach().cpu().numpy())
inference_time = time.perf_counter() - t1

probs  = np.concatenate(probs) if len(probs)>0 else np.zeros((len(y_test),), dtype=float)
preds  = np.concatenate(preds)
labels = np.concatenate(labels)

acc = accuracy_score(labels, preds)
prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary" if n_classes==2 else "macro", zero_division=0)

roc_auc = None
if n_classes == 2 and len(np.unique(labels)) == 2:
    try:
        roc_auc = roc_auc_score(labels, probs)
        fpr, tpr, _ = roc_curve(labels, probs)
        plt.figure(); plt.plot(fpr, tpr); plt.plot([0,1],[0,1],"--"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve")
        plt.savefig(os.path.join(OUT_DIR, "roc_curve.png"), dpi=140); plt.close()
    except Exception:
        roc_auc = None

cm = confusion_matrix(labels, preds)
np.savetxt(os.path.join(OUT_DIR, "confusion_matrix.csv"), cm, fmt="%d", delimiter=",")

avg_latency_ms = (inference_time / max(1, len(labels))) * 1000.0
metrics = {
    "dataset_path": IN_PATH,
    "num_samples": int(len(y)),
    "image_shape_raw": list(X_raw.shape),
    "image_shape_used_nchw": [int(X_nchw.shape[0]), int(C), int(H), int(W)],
    "train_time_sec": round(train_time, 4),
    "inference_total_sec": round(inference_time, 4),
    "inference_avg_ms_per_sample": round(avg_latency_ms, 4),
    "rss_mb_before": round(rss_before, 2),
    "rss_mb_after": round(rss_after, 2),
    "gpu_peak_mb": round(gpu_peak_mb, 2) if gpu_peak_mb is not None else None,
    "accuracy": round(acc, 4),
    "precision": round(prec, 4),
    "recall": round(rec, 4),
    "f1": round(f1, 4),
    "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
    "classes": {int(c): int(n) for c, n in zip(*np.unique(y_train, return_counts=True))}
}
with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print(json.dumps(metrics, indent=2))
