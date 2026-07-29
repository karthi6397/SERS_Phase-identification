# ======================================================
# FINAL MODEL - PHASE IDENTIFICATION (BEST PARAMETERS)
# ======================================================
import numpy as np
import pandas as pd
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -------------------------
# Preprocessing
# -------------------------
def snv(x):
    mean = np.mean(x, axis=1, keepdims=True)
    std = np.std(x, axis=1, keepdims=True) + 1e-8
    return (x - mean) / std


def assign_phase(day):
    if day <= 8:
        return 0  # Lag
    elif day <= 20:
        return 1  # Log
    else:
        return 2  # Stationary


# -------------------------
# Load & Build Dataset (Memory Safe)
# -------------------------
df = pd.read_csv("Merged_1_PL.csv")
samples, labels = [], []
for col in df.columns:
    match = re.match(r"Day(\d+)_Sample\d+", col)
    if match:
        day = int(match.group(1))
        spectrum = df[col].values.astype(np.float32)
        # Savitzky-Golay (1st derivative)
        spectrum = savgol_filter(
            spectrum,
            window_length=25,  # ODD & < spectral length
            polyorder=2,
            deriv=1
        )
        samples.append(spectrum)
        labels.append(assign_phase(day))

X = snv(np.stack(samples)).astype(np.float32)
y = np.array(labels)

print("Dataset shape:", X.shape)
print("Labels:", np.unique(y, return_counts=True))

# -------------------------
# Train / Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train = torch.tensor(X_train).unsqueeze(1)
X_test = torch.tensor(X_test).unsqueeze(1)
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)

train_ds = TensorDataset(X_train, y_train)
test_ds = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)

SEQ_LEN = X.shape[1]


# ======================================================
# TRANSFORMER MODEL (BEST PARAMETERS)
# ======================================================
class PhaseTransformer1D(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.embedding = nn.Linear(seq_len, 64)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        x = x.squeeze(1)                        # (B, L)
        x = self.embedding(x).unsqueeze(1)      # (B, 1, 64)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)


# -------------------------
# Initialize Model
# -------------------------
model = PhaseTransformer1D(SEQ_LEN).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

# ======================================================
# FINAL TRAINING
# ======================================================
EPOCHS = 50
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    acc = correct / total

    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS}] | "
        f"Loss: {train_loss:.4f} | Test Acc: {acc:.4f}"
    )

print("\nFinal training completed.")

# ======================================================
# FINAL MODEL - PHASE IDENTIFICATION
# COMPREHENSIVE EVALUATION PIPELINE
# ======================================================
import re, time, os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, roc_curve,
    log_loss, cohen_kappa_score, brier_score_loss
)
from sklearn.preprocessing import label_binarize
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import psutil

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -------------------------
# Preprocessing
# -------------------------
def snv(x):
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)


def assign_phase(day):
    if day <= 8:
        return 0
    elif day <= 20:
        return 1
    else:
        return 2


# -------------------------
# Load Dataset
# -------------------------
df = pd.read_csv("Merged_1_PL.csv")
samples, labels = [], []
for col in df.columns:
    m = re.match(r"Day(\d+)_Sample\d+", col)
    if m:
        spectrum = savgol_filter(
            df[col].values.astype(np.float32),
            window_length=25, polyorder=2, deriv=1
        )
        samples.append(spectrum)
        labels.append(assign_phase(int(m.group(1))))

X = snv(np.stack(samples)).astype(np.float32)
y = np.array(labels)
n_classes = len(np.unique(y))
SEQ_LEN = X.shape[1]


# ======================================================
# MODEL
# ======================================================
class PhaseTransformer1D(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.embedding = nn.Linear(seq_len, 64)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.fc = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 3)
        )

    def forward(self, x):
        x = self.embedding(x.squeeze(1)).unsqueeze(1)
        x = self.encoder(x).mean(dim=1)
        return self.fc(x)


# ======================================================
# TRAIN + EVALUATE FUNCTION
# ======================================================
def train_and_evaluate(X, y, fold_id="Holdout"):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    train_ds = TensorDataset(torch.tensor(Xtr).unsqueeze(1), torch.tensor(ytr))
    test_ds = TensorDataset(torch.tensor(Xte).unsqueeze(1), torch.tensor(yte))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)

    model = PhaseTransformer1D(SEQ_LEN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()

    best_loss = np.inf
    early_stop_epoch = None
    start_train = time.time()

    for epoch in range(50):
        model.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch_loss < best_loss:
            best_loss = epoch_loss
        elif early_stop_epoch is None:
            early_stop_epoch = epoch + 1

    train_time = time.time() - start_train

    # -------------------------
    # Evaluation
    # -------------------------
    model.eval()
    start_test = time.time()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            probs = torch.softmax(model(xb), dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            y_true.extend(yb.numpy())
            y_pred.extend(preds)
            y_prob.extend(probs)
    test_time = time.time() - start_test

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # -------------------------
    # Metrics
    # -------------------------
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )
    prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro"
    )
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )
    cm = confusion_matrix(y_true, y_pred)
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    auc_macro = roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
    ll = log_loss(y_true, y_prob)
    brier = np.mean([brier_score_loss(y_bin[:, i], y_prob[:, i]) for i in range(3)])
    kappa = cohen_kappa_score(y_true, y_pred)
    mem_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    infer_time = test_time / len(y_true)

    return {
        "fold": fold_id,
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "auc_macro": auc_macro,
        "log_loss": ll,
        "brier": brier,
        "kappa": kappa,
        "train_time_s": train_time,
        "test_time_s": test_time,
        "inference_time_per_sample_s": infer_time,
        "memory_MB": mem_mb,
        "early_stop_epoch": early_stop_epoch,
        "confusion_matrix": cm
    }


# ======================================================
# K-FOLD ROBUSTNESS
# ======================================================
results = []
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for i, (tr, te) in enumerate(kf.split(X, y)):
    res = train_and_evaluate(X[tr], y[tr], fold_id=f"Fold-{i + 1}")
    results.append(res)

df_results = pd.DataFrame(results)

# ======================================================
# SAVE RESULTS
# ======================================================
with pd.ExcelWriter("Transformer_Evaluation.xlsx", engine="openpyxl") as writer:
    df_results.drop(columns=["confusion_matrix"]).to_excel(
        writer, sheet_name="Overall_Metrics", index=False
    )
    cm_all = {r["fold"]: r["confusion_matrix"].flatten() for r in results}
    pd.DataFrame(cm_all).T.to_excel(writer, sheet_name="Confusion_Matrix")
    df_results[["auc_macro", "log_loss", "brier", "kappa"]].to_excel(
        writer, sheet_name="Probabilistic_Agreement"
    )
    df_results[
        ["train_time_s", "test_time_s", "inference_time_per_sample_s", "memory_MB"]
    ].to_excel(writer, sheet_name="Computational_Metrics")

print("\nEvaluation complete")
print("Results saved to Transformer_Evaluation.xlsx")
