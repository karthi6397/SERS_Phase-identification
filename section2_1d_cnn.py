# Segment:1 Reproducibility
import os, random, numpy as np, torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass
os.environ["PYTHONHASHSEED"] = str(SEED)
print(f"Reproducibility enabled with SEED = {SEED}")

# ============================================================
# PHASE IDENTIFICATION CNN
# ============================================================
import pandas as pd
import re, time, psutil
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, confusion_matrix, roc_curve
)
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# -------------------------------
# 1. LOAD & PREPARE DATA
# -------------------------------
df = pd.read_csv("Merged_1_PL.csv")
df_renamed = df.copy()

day_columns = {}
for i in range(37):
    for col in df.columns:
        if re.match(rf"Day{i}_Sample\d+", col):
            day_columns[col] = str(i)
df_renamed.rename(columns=day_columns, inplace=True)


def assign_phase(day):
    if day <= 8:
        return 0
    elif day <= 20:
        return 1
    else:
        return 2


labels_lag, labels_log, labels_stationary = set(), set(), set()
for col in df_renamed.columns:
    if col.isdigit():
        p = assign_phase(int(col))
        (labels_lag if p == 0 else labels_log if p == 1 else labels_stationary).add(col)

df_lag = df_renamed[list(labels_lag)].T.assign(label=0)
df_log = df_renamed[list(labels_log)].T.assign(label=1)
df_stat = df_renamed[list(labels_stationary)].T.assign(label=2)
df_phase = pd.concat([df_lag, df_log, df_stat], ignore_index=True)

X = df_phase.drop(columns="label").values.astype(np.float32)
y = df_phase["label"].values


# -------------------------------
# 2. PREPROCESSING (SG + SNV)
# -------------------------------
def snv(x):
    return (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)


X = savgol_filter(X, window_length=25, polyorder=2, deriv=1, axis=1)
X = snv(X)


# -------------------------------
# 3. MODEL
# -------------------------------
class PhaseCNN(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, 5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)
        self.fc1 = nn.Linear((n_features // 4) * 64, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc2(self.drop(self.relu(self.fc1(x))))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 4. TRAIN / VAL / TEST (70/15/15)
# -------------------------------
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=42
)

X_train = torch.tensor(X_train).unsqueeze(1).to(device)
X_val = torch.tensor(X_val).unsqueeze(1).to(device)
X_test = torch.tensor(X_test).unsqueeze(1).to(device)
y_train = torch.tensor(y_train).to(device)
y_val = torch.tensor(y_val).to(device)
y_test = torch.tensor(y_test).to(device)

model = PhaseCNN(X.shape[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# -------------------------------
# FULL EPOCH LOGGING CONTAINERS
# -------------------------------
epoch_train_loss = []
epoch_train_acc = []  # NEW
epoch_val_loss = []
epoch_val_acc = []
best_val_loss = np.inf
wait, early_epoch = 0, None

start_train = time.time()
for epoch in tqdm(range(50), desc="Training epochs"):
    model.train()
    train_loss = 0.0
    for i in range(0, len(X_train), 32):
        xb, yb = X_train[i:i + 32], y_train[i:i + 32]
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # ---- TRAIN ACCURACY (NEW, minimal)
    with torch.no_grad():
        train_logits_epoch = model(X_train)
        train_acc_epoch = accuracy_score(
            y_train.cpu(), train_logits_epoch.argmax(1).cpu()
        )

    # Validation
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val)
        val_loss = criterion(val_logits, y_val).item()
        val_acc = accuracy_score(y_val.cpu(), val_logits.argmax(1).cpu())

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        wait = 0
    else:
        wait += 1
        if wait == 5 and early_epoch is None:
            early_epoch = epoch + 1

    # ---- LOG EACH EPOCH
    epoch_train_loss.append(train_loss)
    epoch_train_acc.append(train_acc_epoch)  # NEW
    epoch_val_loss.append(val_loss)
    epoch_val_acc.append(val_acc)

train_time = time.time() - start_train
print(f"Early stopping would occur at epoch: {early_epoch}")
print(f"Final Validation Accuracy: {val_acc:.4f}")

# ============================================================
# PUBLICATION EXPORT - ONLY 3 COLUMNS (MODIFIED)
# ============================================================
df_epochs = pd.DataFrame({
    "Epoch": range(1, len(epoch_train_loss) + 1),
    "Train_Accuracy": epoch_train_acc,
    "Train_Loss": epoch_train_loss
})
df_epochs.to_excel("Epoch_TrainAccuracy_TrainLoss_70_15_15.xlsx", index=False)
print("Epoch Excel (3 columns) exported")

# ============================================================
# TEST METRICS (UNCHANGED)
# ============================================================
with torch.no_grad():
    train_probs = torch.softmax(model(X_train), dim=1).cpu().numpy()
    val_probs = torch.softmax(model(X_val), dim=1).cpu().numpy()
    test_probs = torch.softmax(model(X_test), dim=1).cpu().numpy()

train_preds = train_probs.argmax(1)
val_preds = val_probs.argmax(1)
test_preds = test_probs.argmax(1)

acc = accuracy_score(y_test.cpu(), test_preds)
prec = precision_score(y_test.cpu(), test_preds, average="macro")
rec = recall_score(y_test.cpu(), test_preds, average="macro")
f1 = f1_score(y_test.cpu(), test_preds, average="macro")
kappa = cohen_kappa_score(y_test.cpu(), test_preds)

print(f"Test Accuracy: {acc:.4f}")
print(f"Validation Accuracy: {accuracy_score(y_val.cpu(), val_preds):.4f}")

# ============================================================
# ROC EXPORT (UNCHANGED)
# ============================================================
roc_rows = []
for split, ytrue, probs in [
    ("Train", y_train.cpu().numpy(), train_probs),
    ("Validation", y_val.cpu().numpy(), val_probs),
    ("Test", y_test.cpu().numpy(), test_probs)
]:
    for c in range(3):
        fpr, tpr, _ = roc_curve((ytrue == c).astype(int), probs[:, c])
        roc_rows.append(pd.DataFrame({"Split": split, "Class": c, "FPR": fpr, "TPR": tpr}))

pd.concat(roc_rows).to_excel("ROC_Data_Train_Val_Test.xlsx", index=False)

# ============================================================
# CONFUSION MATRIX (UNCHANGED)
# ============================================================
cm = confusion_matrix(y_test.cpu(), test_preds)
labels = ["Lag", "Log", "Stationary"]

plt.rcParams['font.family'] = 'Book Antiqua'
plt.figure(figsize=(8, 7))
sns.heatmap(
    cm, annot=True, fmt='d',
    cmap=sns.light_palette("#AB7100", as_cmap=True),
    annot_kws={"color": "black", "size": 14},
    xticklabels=labels,
    yticklabels=labels
)
plt.xlabel("Predicted", fontsize=16)
plt.ylabel("Actual", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig("ConfusionMatrix_70_15_15.png", dpi=300)
plt.close()

print("\nFINAL PUBLICATION OUTPUTS GENERATED")

# ============================================================
# Step-70_15_15 : TRAIN / VAL / TEST METRICS EXPORT
# ============================================================
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score
)
import time
import psutil

process = psutil.Process()
metrics_rows = []


def collect_metrics(split_name, y_true, y_pred, elapsed_time):
    return {
        "Split": split_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro"),
        "Recall": recall_score(y_true, y_pred, average="macro"),
        "F1_score": f1_score(y_true, y_pred, average="macro"),
        "Kappa": cohen_kappa_score(y_true, y_pred),
        "TestTime_s": elapsed_time,
        "InferenceTime_s": elapsed_time / len(y_pred),
        "Memory_MB": process.memory_info().rss / (1024 ** 2)
    }


# -------------------------------
# TRAIN METRICS
# -------------------------------
start = time.time()
train_preds = train_probs.argmax(1)
train_time_eval = time.time() - start
metrics_rows.append(
    collect_metrics("Train", y_train.cpu().numpy(), train_preds, train_time_eval)
)

# -------------------------------
# VALIDATION METRICS
# -------------------------------
start = time.time()
val_preds = val_probs.argmax(1)
val_time_eval = time.time() - start
metrics_rows.append(
    collect_metrics("Validation", y_val.cpu().numpy(), val_preds, val_time_eval)
)

# -------------------------------
# TEST METRICS
# -------------------------------
start = time.time()
test_preds = test_probs.argmax(1)
test_time_eval = time.time() - start
metrics_rows.append(
    collect_metrics("Test", y_test.cpu().numpy(), test_preds, test_time_eval)
)

# -------------------------------
# SAVE TO EXCEL
# -------------------------------
df_70_15_15 = pd.DataFrame(metrics_rows)
df_70_15_15.insert(5, "TrainTime_s", train_time)  # same training time for all splits
df_70_15_15.to_excel("Metrics_70_15_15_Train_Val_Test.xlsx", index=False)
print("Step-70_15_15 metrics exported successfully")

# ============================================================
# EXTENSION: MULTIPLE TRAIN-TEST RATIOS (MEMORY SAFE)
# ============================================================
from torch.utils.data import DataLoader, TensorDataset
import gc
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    cohen_kappa_score
)

# -------------------------------
# Use already-preprocessed data
# -------------------------------
try:
    X_data = X_processed  # preferred
except NameError:
    X_data = X  # fallback
y_data = y

# Ensure float32 to reduce memory
if X_data.dtype != np.float32:
    X_data = X_data.astype(np.float32)

ratios = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9]
process = psutil.Process()

for r in ratios:
    print(f"\nRunning ratio {int(r * 100)}:{int((1 - r) * 100)}")

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=1 - r,
        random_state=42
    )
    train_idx, test_idx = next(splitter.split(X_data, y_data))

    # Create tensors ONLY for this ratio
    X_train = torch.from_numpy(X_data[train_idx]).unsqueeze(1)
    X_test = torch.from_numpy(X_data[test_idx]).unsqueeze(1)
    y_train = torch.from_numpy(y_data[train_idx]).long()
    y_test = torch.from_numpy(y_data[test_idx]).long()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhaseCNN(X_train.shape[2]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=32,
        shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=32,
        shuffle=False
    )

    # -------------------------------
    # Training
    # -------------------------------
    start_train = time.time()
    for _ in range(20):  # keep light, consistent
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
    train_time = time.time() - start_train

    # -------------------------------
    # Testing
    # -------------------------------
    start_test = time.time()
    model.eval()
    y_pred = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            y_pred.extend(torch.argmax(model(xb), dim=1).cpu().numpy())
    test_time = time.time() - start_test
    inference_time = test_time / len(y_pred)

    # -------------------------------
    # Metrics
    # -------------------------------
    acc = accuracy_score(y_test.numpy(), y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test.numpy(), y_pred, average="macro"
    )
    kappa = cohen_kappa_score(y_test.numpy(), y_pred)
    mem_mb = process.memory_info().rss / (1024 ** 2)

    # -------------------------------
    # Save results
    # -------------------------------
    df_metrics = pd.DataFrame([{
        "Train_%": int(r * 100),
        "Test_%": int((1 - r) * 100),
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1_score": f1,
        "Kappa": kappa,
        "TrainTime_s": train_time,
        "TestTime_s": test_time,
        "InferenceTime_s": inference_time,
        "Memory_MB": mem_mb
    }])
    out_name = f"Metrics_{int(r * 100)}_{int((1 - r) * 100)}.xlsx"
    df_metrics.to_excel(out_name, index=False)
    print(f"Saved: {out_name}")

    # -------------------------------
    # Cleanup (CRITICAL)
    # -------------------------------
    del X_train, X_test, y_train, y_test, model
    torch.cuda.empty_cache()
    gc.collect()

print("\nEXTENSION COMPLETED: All ratios processed successfully")
