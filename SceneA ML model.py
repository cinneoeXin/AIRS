import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from time import time
import matplotlib.pyplot as plt
import tracemalloc
import gc

# 1. Data Loading
DATA_PATH = r"E:\USA\AIRS\AIRS WEEK\WEEK10\matched_columns_file_TII_B.xlsx"
df = pd.read_excel(DATA_PATH)

# 2. Label Binarization Encoding：Benign->0, 其它->1
df['Label_bin'] = (df['Label'] != 'Benign').astype(int)

# 3. Only retain numeric feature columns (automatically remove string columns, such as Timestamp/Label）
features = df.select_dtypes(include=[np.number])

# 4. Retrieve Tags
labels = df['Label_bin']

# 5. Standardized Processing
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 6. Dataset Splitting（7:1.5:1.5）
X_train, X_temp, y_train, y_temp = train_test_split(
    features_scaled, labels, test_size=0.3, random_state=42, stratify=labels)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

X_tr, y_tr = X_train, y_train
X_vali, y_vali = X_val, y_val
X_te, y_te = X_test, y_test

# 7. Define Model
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
    model.fit(X_tr, y_tr)
    train_time = time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Inference Latency Measurement (Single-Sample Average, First 100 Samples)
    inference_times = []
    for i in range(min(100, X_te.shape[0])):
        t0 = time()
        _ = model.predict(X_te[i:i+1])
        inference_times.append((time() - t0) * 1000)  # ms

    y_pred = model.predict(X_te)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_te)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_te)
        if y_score.ndim > 1:
            y_score = y_score[:, 1]
    else:
        y_score = y_pred  # fallback

    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average='binary', zero_division=0)
    rec = recall_score(y_te, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_te, y_pred, average='binary', zero_division=0)
    try:
        rocauc = roc_auc_score(y_te, y_score)
    except:
        rocauc = np.nan
    infer_latency = np.mean(inference_times)
    peak_mem = peak / (1024 * 1024)  # bytes -> MB

    results[name] = [acc, prec, rec, f1, rocauc, train_time, infer_latency, peak_mem]

# 8. Results Visualization
results_df = pd.DataFrame(results, index=bar_metrics)
print(results_df)

plt.figure(figsize=(14, 8))
for idx, metric in enumerate(bar_metrics[:5]):  # The first five are classification metrics.
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

# 9. Save Results
results_df.to_csv("traditional_ml_sceneA_results.csv")
