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

# ========== 文件路径 ==========
path_bccc = r"E:\USA\AIRS\AIRS WEEK\WEEK10\matched_columns_file_BCCC_B.xlsx"
path_tii  = r"E:\USA\AIRS\AIRS WEEK\WEEK10\matched_columns_file_TII_B.xlsx"
path_net  = r"E:\USA\AIRS\AIRS WEEK\WEEK10\Network_dataset_1.xlsx"

# ========== 读取数据 ==========
df_bccc = pd.read_excel(path_bccc)
df_tii  = pd.read_excel(path_tii)
df_net  = pd.read_excel(path_net)

# ========== 对齐字段名 ==========
df_bccc.rename(columns={'src_port':'src_port', 'dst_port':'dst_port', 'duration':'duration'}, inplace=True)
df_tii.rename(columns={'Src Port':'src_port', 'Dst Port':'dst_port', 'Flow Duration':'duration'}, inplace=True)
df_net.rename(columns={'src_port':'src_port', 'dst_port':'dst_port', 'duration':'duration'}, inplace=True)

# ========== 只保留交集特征 ==========
common_features = ['src_port', 'dst_port', 'duration']
for col in common_features:
    for df in [df_bccc, df_tii, df_net]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# ========== 标签标准化 ==========
df_bccc['label_bin'] = (df_bccc['label'] != 'Non-Encrypted').astype(int)
df_tii['label_bin']  = (df_tii['Label'] != 'Benign').astype(int)
df_net['label_bin']  = (df_net['type'] != 'normal').astype(int)

# ========== 合并数据 ==========
features = pd.concat([
    df_bccc[common_features],
    df_tii[common_features],
    df_net[common_features]
], axis=0, ignore_index=True)

labels = pd.concat([
    df_bccc['label_bin'],
    df_tii['label_bin'],
    df_net['label_bin']
], axis=0, ignore_index=True)

# ========== 数据分割 ==========
X_train, X_temp, y_train, y_temp = train_test_split(
    features, labels, test_size=0.3, random_state=42, stratify=labels)
X_test, X_val, y_test, y_val = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# ========== 标准化 ==========
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
# X_val_scaled   = scaler.transform(X_val) # 可以注释掉，没用到

# ========== 并行训练+加速SVM ==========
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    # SVM用linear加速，适合特征数较少场景
    "SVM": SVC(kernel='linear', probability=True, random_state=42, max_iter=1000),
    "Logistic Regression": LogisticRegression(max_iter=200, random_state=42, n_jobs=-1),
    "kNN": KNeighborsClassifier(n_neighbors=3, n_jobs=-1), # n_neighbors小些更快
}

results = {}
bar_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC', 'Train Time (s)', 'Inference Latency (ms)', 'Peak Memory (MB)']

for name, model in models.items():
    print(f"\nTraining {name}...")
    gc.collect()
    tracemalloc.start()
    start_time = time()
    model.fit(X_train_scaled, y_train)
    train_time = time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    inference_times = []
    for i in range(min(30, X_test_scaled.shape[0])):  # 只取30个测试样本计算推理延迟
        t0 = time()
        _ = model.predict(X_test_scaled[i:i+1])
        inference_times.append((time() - t0) * 1000)
    infer_latency = np.mean(inference_times)

    y_pred = model.predict(X_test_scaled)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test_scaled)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test_scaled)
        if y_score.ndim > 1:
            y_score = y_score[:, 1]
    else:
        y_score = y_pred

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='binary', zero_division=0)
    rec = recall_score(y_test, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
    try:
        rocauc = roc_auc_score(y_test, y_score)
    except:
        rocauc = np.nan

    peak_mem = peak / (1024 * 1024)
    results[name] = [acc, prec, rec, f1, rocauc, train_time, infer_latency, peak_mem]

    del model
    gc.collect()

results_df = pd.DataFrame(results, index=bar_metrics)
print(results_df)

plt.figure(figsize=(12, 6))
for idx, metric in enumerate(bar_metrics[:5]):
    plt.subplot(2, 3, idx+1)
    plt.bar(results_df.columns, results_df.loc[metric], color='steelblue')
    plt.title(metric)
    plt.ylim(0, 1 if metric != "Train Time (s)" else max(results_df.loc[metric]) * 1.2)
    plt.xticks(rotation=20)
    plt.ylabel(metric)
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 3))
for idx, metric in enumerate(bar_metrics[5:]):
    plt.subplot(1, 3, idx+1)
    plt.bar(results_df.columns, results_df.loc[metric], color='coral')
    plt.title(metric)
    plt.xticks(rotation=20)
    plt.ylabel(metric)
plt.tight_layout()
plt.show()

results_df.to_csv("traditional_ml_sceneC_results.csv")
