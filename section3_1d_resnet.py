# ==========================================================
# REPRODUCIBILITY BLOCK
# ==========================================================
import random, numpy as np, torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

# ==========================================================
# FINAL MODEL: ResNet-1D Phase Identification (Raman)
# ==========================================================
import pandas as pd
import re
import time
import psutil
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import seaborn as sns  # NOTE (fixed): moved up from later in the original
                        # notebook, where it was imported after its first use

DEVICE = torch.device("cpu")


# ==========================================================
# SNV
# ==========================================================
def snv(x):
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)


# ==========================================================
# Phase assignment
# ==========================================================
def assign_phase(day):
    if day <= 8:
        return 0
    elif day <= 20:
        return 1
    else:
        return 2


# ==========================================================
# MEMORY SAFE DATA BUILD
# ==========================================================
df = pd.read_csv("Merged_1_PL.csv")
df_renamed = df.copy()

day_columns = {}
for i in range(37):
    for col in df.columns:
        if re.match(rf"Day{i}_Sample\d+", col):
            day_columns[col] = str(i)
df_renamed.rename(columns=day_columns, inplace=True)

labels_lag, labels_log, labels_stat = set(), set(), set()
for col in df_renamed.columns:
    if col.isdigit():
        p = assign_phase(int(col))
        (labels_lag if p == 0 else labels_log if p == 1 else labels_stat).add(col)

df_lag = df_renamed[list(labels_lag)].T.assign(label=0)
df_log = df_renamed[list(labels_log)].T.assign(label=1)
df_stat = df_renamed[list(labels_stat)].T.assign(label=2)
df_phase = pd.concat([df_lag, df_log, df_stat], ignore_index=True)

X = df_phase.drop(columns="label").values.astype(np.float32)
y = df_phase["label"].values

# ==========================================================
# Spectral preprocessing
# ==========================================================
X = savgol_filter(X, 25, 2, deriv=1, axis=1)
X = snv(X)
X = StandardScaler().fit_transform(X).astype(np.float32)

# ==========================================================
# SPLIT
# ==========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)
X_train = torch.tensor(X_train).unsqueeze(1)
y_train = torch.tensor(y_train)

g = torch.Generator().manual_seed(SEED)
train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=16,
    shuffle=True,
    generator=g
)


# ==========================================================
# ResNet1D
# ==========================================================
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, k, s):
        super().__init__()
        p = k // 2
        self.conv1 = nn.Conv1d(in_c, out_c, k, s, p)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, k, 1, p)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.skip = nn.Sequential()
        if s != 1 or in_c != out_c:
            self.skip = nn.Sequential(
                nn.Conv1d(in_c, out_c, 1, s),
                nn.BatchNorm1d(out_c)
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.skip(x))


class ResNet1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = ResidualBlock(1, 32, 5, 2)
        self.b2 = ResidualBlock(32, 64, 5, 2)
        self.b3 = ResidualBlock(64, 128, 3, 2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, 3)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


model = ResNet1D().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-4)

# ==========================================================
# TRAIN + RECORD EPOCH METRICS
# ==========================================================
best_loss = np.inf
patience = 8
wait = 0
early_epoch = None
epoch_losses = []
epoch_accuracies = []

start_train = time.time()
for epoch in range(50):
    model.train()
    loss_sum = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    avg_loss = loss_sum / len(train_loader)

    # ---- Train accuracy ----
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in train_loader:
            preds = torch.argmax(model(xb), 1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    epoch_acc = correct / total

    epoch_losses.append(avg_loss)
    epoch_accuracies.append(epoch_acc)
    print(f"Epoch {epoch + 1}/50 Loss {avg_loss:.4f} Acc {epoch_acc:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        wait = 0
    else:
        wait += 1
        if wait == patience and early_epoch is None:
            early_epoch = epoch + 1
            print(f"Early stopping triggered at epoch {early_epoch}")

train_time = time.time() - start_train

# ==========================================================
# SAVE EPOCH DATA + PLOT (WITH EARLY STOP MARK)
# ==========================================================
epochs = np.arange(1, 51)
pd.DataFrame({
    "Epoch": epochs,
    "Train_Loss": epoch_losses,
    "Train_Accuracy": epoch_accuracies,
    "Early_Stop_Epoch": [early_epoch] * 50
}).to_excel("Epochwise_Accuracy_Loss.xlsx", index=False)

plt.figure(figsize=(8, 5))
plt.plot(epochs, epoch_losses, label="Train Loss")
plt.plot(epochs, epoch_accuracies, label="Train Accuracy")

# ---- Early stop vertical line ----
if early_epoch is not None:
    plt.axvline(early_epoch, linestyle="--", label=f"Early Stop @ {early_epoch}")

plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Training Accuracy vs Loss per Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Accuracy_vs_Loss_Epoch.png", dpi=300)
plt.close()

print("Saved -> Accuracy_vs_Loss_Epoch.png")
print("Saved -> Epochwise_Accuracy_Loss.xlsx")
print(f"Early stopping occurred at epoch: {early_epoch}")

# ==========================================================
# TEMPLATE METRICS (FIXED TENSOR BUG)
# ==========================================================
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, cohen_kappa_score, confusion_matrix
)

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


# --- ensure tensor ---
X_test_tensor = torch.tensor(X_test).unsqueeze(1).float()
y_test_tensor = torch.tensor(y_test)

# TEST
start = time.time()
test_probs = torch.softmax(model(X_test_tensor), dim=1).detach().cpu().numpy()
test_preds = test_probs.argmax(1)
test_time_eval = time.time() - start
metrics_rows.append(
    collect_metrics("Test", y_test, test_preds, test_time_eval)
)

pd.DataFrame(metrics_rows).to_excel("Metrics_Test.xlsx", index=False)

# ==========================================================
# CONFUSION MATRIX (PUBLICATION STYLE)
# ==========================================================
cm_test = confusion_matrix(y_test, test_preds)
labels = ["Lag", "Log", "Stationary"]

plt.figure(figsize=(8, 7))
sns.heatmap(
    cm_test, annot=True, fmt='d',
    cmap=sns.light_palette("#AB7100", as_cmap=True),
    annot_kws={"color": "black", "size": 14},
    xticklabels=labels, yticklabels=labels
)
plt.xlabel("Predicted", fontsize=16)
plt.ylabel("Actual", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig("ConfusionMatrix_Test.png", dpi=300)
plt.close()

# ==========================================================
# EXTENSION: 70:15:15 SPLIT + FINAL ACCURACY + TEST CM
# (No modification to existing pipeline)
# ==========================================================
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from torch.utils.data import DataLoader, TensorDataset

# -----------------------------
# 70 : 15 : 15 split
# -----------------------------
X_train70, X_temp, y_train70, y_temp = train_test_split(
    X, y, stratify=y, test_size=0.30, random_state=42
)
X_val70, X_test70, y_val70, y_test70 = train_test_split(
    X_temp, y_temp, stratify=y_temp, test_size=0.50, random_state=42
)

# tensors
X_train70_t = torch.tensor(X_train70).unsqueeze(1)
X_val70_t = torch.tensor(X_val70).unsqueeze(1)
X_test70_t = torch.tensor(X_test70).unsqueeze(1)
y_train70_t = torch.tensor(y_train70)
y_val70_t = torch.tensor(y_val70)
y_test70_t = torch.tensor(y_test70)

# loaders
train70_loader = DataLoader(TensorDataset(X_train70_t, y_train70_t), batch_size=16)
val70_loader = DataLoader(TensorDataset(X_val70_t, y_val70_t), batch_size=16)
test70_loader = DataLoader(TensorDataset(X_test70_t, y_test70_t), batch_size=16)


# -----------------------------
# accuracy helper
# -----------------------------
def compute_acc(loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            preds = torch.argmax(model(xb), 1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return 100 * correct / total


train_acc70 = compute_acc(train70_loader)
val_acc70 = compute_acc(val70_loader)
test_acc70 = compute_acc(test70_loader)

print(f"\nFinal train Accuracy: {train_acc70:.2f}% | Test Accuracy: {test_acc70:.2f}% | Validation Accuracy: {val_acc70:.2f}%")

# -----------------------------
# TEST CONFUSION MATRIX (Publication Font Safe)
# -----------------------------
import matplotlib.font_manager as fm

# ---- ensure Book Antiqua availability ----
available_fonts = {f.name for f in fm.fontManager.ttflist}
if "Book Antiqua" in available_fonts:
    plt.rcParams["font.family"] = "Book Antiqua"
else:
    plt.rcParams["font.family"] = "Times New Roman"  # safe journal fallback

# NOTE (fixed): y_true_cm / y_pred_cm were referenced without being defined
# in the original notebook. They are collected here from the 70:15:15 test
# split so the confusion matrix reflects that split, consistent with the
# train/val/test accuracies computed above.
y_true_cm, y_pred_cm = [], []
model.eval()
with torch.no_grad():
    for xb, yb in test70_loader:
        preds = torch.argmax(model(xb), 1)
        y_true_cm.extend(yb.numpy())
        y_pred_cm.extend(preds.numpy())

cm_70 = confusion_matrix(y_true_cm, y_pred_cm)
class_names = ["Lag", "Log", "Stationary"]

plt.figure(figsize=(8, 7))
sns.heatmap(
    cm_70,
    annot=True,
    fmt="d",
    cmap=sns.light_palette("#AB7100", as_cmap=True),
    annot_kws={"color": "black", "size": 14},
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.xlabel("Predicted", fontsize=16)
plt.ylabel("Actual", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig("Confusion_Matrix_70_15_15.png", dpi=300)
plt.close()
print("Saved: Confusion_Matrix_70_15_15.png")

# ==========================================================
# EXTENSION: Multi Split Ratio Evaluation (No code changes)
# ==========================================================
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

split_ratios = [
    (0.5, 0.5),
    (0.6, 0.4),
    (0.65, 0.35),
    (0.7, 0.3),
    (0.75, 0.25),
    (0.8, 0.2),
    (0.9, 0.1),
]


def train_and_measure(X_np, y_np, train_size):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_np, y_np,
        train_size=train_size,
        stratify=y_np,
        random_state=42
    )
    X_tr = torch.tensor(X_tr).unsqueeze(1)
    X_te = torch.tensor(X_te).unsqueeze(1)
    y_tr = torch.tensor(y_tr)
    y_te = torch.tensor(y_te)

    train_loader_ext = DataLoader(TensorDataset(X_tr, y_tr), batch_size=16, shuffle=True)
    test_loader_ext = DataLoader(TensorDataset(X_te, y_te), batch_size=16)

    model_ext = ResNet1D()
    criterion_ext = nn.CrossEntropyLoss()
    optimizer_ext = optim.Adam(model_ext.parameters(), lr=5e-4)

    # -------- Train --------
    t0 = time.time()
    for _ in range(50):
        model_ext.train()
        for xb, yb in train_loader_ext:
            optimizer_ext.zero_grad()
            loss = criterion_ext(model_ext(xb), yb)
            loss.backward()
            optimizer_ext.step()
    train_time = time.time() - t0

    # -------- Test --------
    y_true = []
    y_pred = []
    t1 = time.time()
    with torch.no_grad():
        for xb, yb in test_loader_ext:
            logits = model_ext(xb)
            preds = logits.argmax(1)
            y_true.extend(yb.numpy())
            y_pred.extend(preds.numpy())
    test_time = time.time() - t1
    infer_time = test_time / len(y_true)

    # -------- Metrics --------
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    kappa = cohen_kappa_score(y_true, y_pred)
    mem_mb = psutil.Process().memory_info().rss / 1024 ** 2

    return {
        "Accuracy": acc,
        "Precision": p,
        "Recall": r,
        "F1": f1,
        "Kappa": kappa,
        "TrainTime_s": train_time,
        "TestTime_s": test_time,
        "InferenceTime_s": infer_time,
        "Memory_MB": mem_mb
    }


# ==========================================================
# RUN ALL RATIOS
# ==========================================================
for tr, te in split_ratios:
    print(f"\nRunning split {int(tr * 100)}:{int(te * 100)}")
    res = train_and_measure(X, y, tr)
    df_out = pd.DataFrame([res])
    fname = f"ResNet1D_split_{int(tr * 100)}_{int(te * 100)}.xlsx"
    df_out.to_excel(fname, index=False)
    print("Saved ->", fname)
