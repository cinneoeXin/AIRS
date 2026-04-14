import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from time import time
import matplotlib.pyplot as plt
import tracemalloc
import gc

# ========== Path Configuration ==========
train_val_path = "E:/matched_columns_file_BCCC_B.xlsx"
test_path = "E:/Network_dataset_1.xlsx"

# ========== Feature Intersection ==========
common_features = ['src_port', 'dst_port', 'duration']

# ========== Training Set ==========
df_tr = pd.read_excel(train_val_path)
# Label binarization—for example, Non-Encrypted = 0 and all others = 1—or a custom classification scheme of your choice.
df_tr['label_bin'] = (df_tr['label'] != 'Non-Encrypted').astype(int)  # Adjust independently based on the task.
features_tr = df_tr[common_features]
labels_tr = df_tr['label_bin']

# ========== Training Set ==========
df_te = pd.read_excel(test_path)
# Here, the `type` field requires custom mapping for binary classification based on the specific task.
df_te['type_bin'] = (df_te['type'] != 'normal').astype(int)  # Normal traffic is 0; anything else is 1.
features_te = df_te[common_features]
labels_te = df_te['type_bin']

# ========== standardization ==========
scaler = StandardScaler()
features_tr_scaled = scaler.fit_transform(features_tr)
features_te_scaled = scaler.transform(features_te)

# ========== Training Set Splitting ==========
X_train, X_val, y_train, y_val = train_test_split(
    features_tr_scaled, labels_tr, test_size=0.15, random_state=42, stratify=labels_tr
)

# ========== model ==========
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
    "kNN": KNeighborsClassifier(n_neighbors=5),
}

results = {}
bar_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC', 'Train Time (s)', 'Inference Latency (ms)', 'Peak Memory (MB)']

for name, model in models.items():
    print(f"Training {name}...")
    gc.collect()
    tracemalloc.start()
    start_time = time()
    model.fit(X_train, y_train)
    train_time = time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Inference Latency Measurement (Single-Sample Average, First 100 Samples)
    inference_times = []
    for i in range(min(100, features_te_scaled.shape[0])):
        t0 = time()
        _ = model.predict(features_te_scaled[i:i+1])
        inference_times.append((time() - t0) * 1000)  # ms

    y_pred = model.predict(features_te_scaled)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(features_te_scaled)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(features_te_scaled)
        if y_score.ndim > 1:
            y_score = y_score[:, 1]
    else:
        y_score = y_pred

    acc = accuracy_score(labels_te, y_pred)
    prec = precision_score(labels_te, y_pred, average='binary', zero_division=0)
    rec = recall_score(labels_te, y_pred, average='binary', zero_division=0)
    f1 = f1_score(labels_te, y_pred, average='binary', zero_division=0)
    try:
        rocauc = roc_auc_score(labels_te, y_score)
    except:
        rocauc = np.nan
    infer_latency = np.mean(inference_times)
    peak_mem = peak / (1024 * 1024)  # bytes -> MB

    results[name] = [acc, prec, rec, f1, rocauc, train_time, infer_latency, peak_mem]

# ========== Results Visualization ==========
results_df = pd.DataFrame(results, index=bar_metrics)
print(results_df)

plt.figure(figsize=(14, 8))
for idx, metric in enumerate(bar_metrics[:5]):
    plt.subplot(2, 3, idx+1)
    plt.bar(results_df.columns, results_df.loc[metric], color='steelblue')
    plt.title(metric)
    plt.ylim(0, 1 if metric != "Train Time (s)" else max(results_df.loc[metric]) * 1.2)
    plt.xticks(rotation=20)
    plt.ylabel(metric)
plt.tight_layout()
plt.show()

# Compute/Resource Metrics
plt.figure(figsize=(12, 4))
for idx, metric in enumerate(bar_metrics[5:]):
    plt.subplot(1, 3, idx+1)
    plt.bar(results_df.columns, results_df.loc[metric], color='coral')
    plt.title(metric)
    plt.xticks(rotation=20)
    plt.ylabel(metric)
plt.tight_layout()
plt.show()

results_df.to_csv("traditional_ml_sceneB_results.csv")
