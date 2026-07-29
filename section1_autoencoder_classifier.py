# ==========================================================
# REPRODUCIBILITY SETTINGS (ADD AT VERY TOP OF SCRIPT)
# ==========================================================
import os
import random
import numpy as np

SEED = 42

# Python hash seed
os.environ["PYTHONHASHSEED"] = str(SEED)

# TensorFlow deterministic behavior (must be set BEFORE TF import)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

# Optional: disable GPU for strict reproducibility
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Set seeds
random.seed(SEED)
np.random.seed(SEED)

print("Reproducibility enabled with SEED =", SEED)

# ==========================================================
# FINAL MODEL + COMPREHENSIVE EVALUATION (70:15:15 SPLIT)
# ==========================================================
import pandas as pd
import re
import tensorflow as tf
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import confusion_matrix, roc_curve, auc
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ==========================================================
# Reproducibility
# ==========================================================
np.random.seed(42)
tf.random.set_seed(42)

# ==========================================================
# SNV
# ==========================================================
def snv(X):
    return (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)

# ==========================================================
# Load + Phase Identification
# ==========================================================
df = pd.read_csv("Merged_1_PL.csv")
df_r = df.copy()

rename = {}
for i in range(37):
    for c in df.columns:
        if re.match(rf"Day{i}_Sample\d+", c):
            rename[c] = str(i)
df_r.rename(columns=rename, inplace=True)


def assign_phase(day):
    if day <= 8:
        return 0
    elif day <= 20:
        return 1
    else:
        return 2


lag, log, stat = set(), set(), set()
for c in df_r.columns:
    if c.isdigit():
        p = assign_phase(int(c))
        (lag if p == 0 else log if p == 1 else stat).add(c)

df_phase = pd.concat([
    df_r[list(lag)].T.assign(label=0),
    df_r[list(log)].T.assign(label=1),
    df_r[list(stat)].T.assign(label=2)
], ignore_index=True)

X = df_phase.drop(columns=["label"]).values
y = df_phase["label"].values

# ==========================================================
# Preprocessing
# ==========================================================
X = savgol_filter(X, 25, 2, deriv=1, axis=1)
X = snv(X)
X = StandardScaler().fit_transform(X)

# ==========================================================
# 70 : 15 : 15 SPLIT
# ==========================================================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

y_train_cat = to_categorical(y_train, 3)
y_val_cat = to_categorical(y_val, 3)
y_test_cat = to_categorical(y_test, 3)

# ==========================================================
# Best Params
# ==========================================================
ENC = 64
DROP = 0.3
BS = 16
LR = 0.001
EPOCHS = 50

tf.keras.backend.clear_session()

# ==========================================================
# Autoencoder
# ==========================================================
inp = Input(shape=(X_train.shape[1],))
x = Dense(256, activation="relu")(inp)
x = BatchNormalization()(x)
x = Dense(128, activation="relu")(x)
x = BatchNormalization()(x)
enc = Dense(ENC, activation="relu")(x)
x = Dense(128, activation="relu")(enc)
x = Dense(256, activation="relu")(x)
out = Dense(X_train.shape[1])(x)

ae = Model(inp, out)
ae.compile(optimizer=Adam(LR), loss="mse")

es = EarlyStopping(patience=6, restore_best_weights=True)

hist_ae = ae.fit(
    X_train, X_train,
    epochs=EPOCHS,
    batch_size=BS,
    validation_data=(X_val, X_val),
    callbacks=[es],
    verbose=0
)
print(f"Autoencoder early stopping at epoch {len(hist_ae.history['loss'])}")

enc_model = Model(inp, enc)
X_train_e = enc_model.predict(X_train, verbose=0)
X_val_e = enc_model.predict(X_val, verbose=0)
X_test_e = enc_model.predict(X_test, verbose=0)

# ==========================================================
# Classifier
# ==========================================================
clf = Sequential([
    Dense(128, activation="relu", input_shape=(ENC,)),
    Dropout(DROP),
    Dense(64, activation="relu"),
    Dropout(DROP),
    Dense(32, activation="relu"),
    Dense(3, activation="softmax")
])
clf.compile(
    optimizer=Adam(LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

hist_clf = clf.fit(
    X_train_e, y_train_cat,
    epochs=EPOCHS,
    batch_size=BS,
    validation_data=(X_val_e, y_val_cat),
    callbacks=[es],
    verbose=0
)
print(f"Classifier early stopping at epoch {len(hist_clf.history['loss'])}")

# ==========================================================
# CONFUSION MATRIX (TEST SET)
# ==========================================================
y_test_prob = clf.predict(X_test_e, verbose=0)
y_test_pred = np.argmax(y_test_prob, axis=1)

cm = confusion_matrix(y_test, y_test_pred)
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
plt.xlabel('Predicted', fontsize=16)
plt.ylabel('Actual', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig("ConfusionMatrix_Test.png", dpi=300)
plt.close()

# ==========================================================
# ROC DATA EXPORT (TRAIN / VAL / TEST)
# ==========================================================
roc_records = []


def collect_roc(split_name, y_true, y_prob):
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    for i, cls in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_records.append({
            "Split": split_name,
            "Class": cls,
            "FPR": fpr.tolist(),
            "TPR": tpr.tolist()
        })


collect_roc("Train", y_train, clf.predict(X_train_e, verbose=0))
collect_roc("Validation", y_val, clf.predict(X_val_e, verbose=0))
collect_roc("Test", y_test, y_test_prob)

roc_df = pd.DataFrame(roc_records)
with pd.ExcelWriter("ROC_70_15_15.xlsx", engine="xlsxwriter") as writer:
    roc_df.to_excel(writer, sheet_name="ROC_Data", index=False)

print("\nCONFIRMED: 70:15:15 split used")
print("ROC data saved as: ROC_70_15_15.xlsx")
print("Confusion matrix saved for TEST set")

# ==========================================================
# TRAINING CURVES EXPORT (0 -> 50 EPOCHS, EXCEL + PNG)
# ==========================================================
MAX_EPOCHS = 50

# --------- Original history ---------
train_loss = hist_clf.history["loss"]
val_loss = hist_clf.history["val_loss"]
train_acc = hist_clf.history["accuracy"]
val_acc = hist_clf.history["val_accuracy"]
actual_epochs = len(train_loss)


# --------- Pad to 50 epochs if early stopping occurred ---------
def pad_to_max(values, max_len):
    if len(values) < max_len:
        return values + [values[-1]] * (max_len - len(values))
    return values[:max_len]


train_loss = pad_to_max(train_loss, MAX_EPOCHS)
val_loss = pad_to_max(val_loss, MAX_EPOCHS)
train_acc = pad_to_max(train_acc, MAX_EPOCHS)
val_acc = pad_to_max(val_acc, MAX_EPOCHS)

epochs = list(range(0, MAX_EPOCHS))  # 0 -> 49 (epoch index style)

# --------- Create DataFrame ---------
df_history = pd.DataFrame({
    "Epoch": epochs,
    "Train_Loss": train_loss,
    "Val_Loss": val_loss,
    "Train_Accuracy": train_acc,
    "Val_Accuracy": val_acc,
})

# --------- Save Excel ---------
excel_path = "Accuracy_Loss_vs_Epoch.xlsx"
df_history.to_excel(excel_path, index=False)
print(f"Excel saved: {excel_path}")

# --------- Plot PNG (same data) ---------
plt.rcParams['font.family'] = 'Book Antiqua'
plt.figure(figsize=(8, 6))
plt.plot(df_history["Epoch"], df_history["Train_Accuracy"], label="Train Accuracy", linewidth=2)
plt.plot(df_history["Epoch"], df_history["Val_Accuracy"], label="Validation Accuracy", linewidth=2)
plt.plot(df_history["Epoch"], df_history["Train_Loss"], "--", label="Train Loss", linewidth=2)
plt.plot(df_history["Epoch"], df_history["Val_Loss"], "--", label="Validation Loss", linewidth=2)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Metric Value", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("Accuracy_Loss_vs_Epoch.png", dpi=300)
plt.close()
print("PNG saved: Accuracy_Loss_vs_Epoch.png")

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
train_probs = clf.predict(X_train_e, verbose=0)
train_preds = np.argmax(train_probs, axis=1)
train_time_eval = time.time() - start
metrics_rows.append(
    collect_metrics("Train", y_train, train_preds, train_time_eval)
)

# -------------------------------
# VALIDATION METRICS
# -------------------------------
start = time.time()
val_probs = clf.predict(X_val_e, verbose=0)
val_preds = np.argmax(val_probs, axis=1)
val_time_eval = time.time() - start
metrics_rows.append(
    collect_metrics("Validation", y_val, val_preds, val_time_eval)
)

# -------------------------------
# TEST METRICS
# -------------------------------
start = time.time()
test_preds = y_test_pred  # already computed earlier
test_time_eval = time.time() - start
metrics_rows.append(
    collect_metrics("Test", y_test, test_preds, test_time_eval)
)

# -------------------------------
# SAVE TO EXCEL
# -------------------------------
df_70_15_15 = pd.DataFrame(metrics_rows)
df_70_15_15.insert(
    5,
    "TrainTime_s",
    hist_clf.history["loss"].__len__()  # same training process, single value
)
df_70_15_15.to_excel("Metrics_70_15_15_Train_Val_Test.xlsx", index=False)
print("Step-70_15_15 metrics exported successfully")

# ==========================================================
# EXTENSION: MULTIPLE TRAIN-TEST RATIOS (NO CODE MODIFICATION)
# ==========================================================
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score

split_ratios = {
    "50_50": 0.50,
    "60_40": 0.40,
    "65_35": 0.35,
    "70_30": 0.30,
    "75_25": 0.25,
    "80_20": 0.20,
    "90_10": 0.10
}

for tag, test_size in split_ratios.items():
    print(f"\nRunning split {tag.replace('_', ':')}")
    tf.keras.backend.clear_session()
    gc.collect()

    # ---------------- Split ----------------
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=42
    )
    y_tr_cat = to_categorical(y_tr, 3)
    y_te_cat = to_categorical(y_te, 3)

    # ---------------- Memory + Time ----------------
    start_mem = process.memory_info().rss / 1024 ** 2
    t0 = time.time()

    # ---------------- Autoencoder ----------------
    inp = Input(shape=(X_tr.shape[1],))
    x = Dense(256, activation="relu")(inp)
    x = BatchNormalization()(x)
    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)
    enc = Dense(ENC, activation="relu")(x)
    x = Dense(128, activation="relu")(enc)
    x = Dense(256, activation="relu")(x)
    out = Dense(X_tr.shape[1])(x)

    ae = Model(inp, out)
    ae.compile(optimizer=Adam(LR), loss="mse")
    ae.fit(
        X_tr, X_tr,
        epochs=EPOCHS,
        batch_size=BS,
        validation_split=0.1,
        callbacks=[EarlyStopping(patience=6, restore_best_weights=True)],
        verbose=0
    )

    enc_model = Model(inp, enc)
    X_tr_e = enc_model.predict(X_tr, verbose=0)
    X_te_e = enc_model.predict(X_te, verbose=0)

    # ---------------- Classifier ----------------
    clf = Sequential([
        Dense(128, activation="relu", input_shape=(ENC,)),
        Dropout(DROP),
        Dense(64, activation="relu"),
        Dropout(DROP),
        Dense(32, activation="relu"),
        Dense(3, activation="softmax")
    ])
    clf.compile(
        optimizer=Adam(LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    clf.fit(
        X_tr_e, y_tr_cat,
        epochs=EPOCHS,
        batch_size=BS,
        validation_split=0.1,
        callbacks=[EarlyStopping(patience=6, restore_best_weights=True)],
        verbose=0
    )
    train_time = time.time() - t0

    # ---------------- Evaluation ----------------
    t1 = time.time()
    y_prob = clf.predict(X_te_e, verbose=0)
    inference_time = (time.time() - t1) / len(y_te)
    y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_te, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_te, y_pred, average="macro")
    kappa = cohen_kappa_score(y_te, y_pred)
    test_time = time.time() - t1
    end_mem = process.memory_info().rss / 1024 ** 2

    # ---------------- Save Results ----------------
    df_result = pd.DataFrame([{
        "Split": tag.replace("_", ":"),
        "Accuracy": acc,
        "Precision_macro": prec,
        "Recall_macro": rec,
        "F1_macro": f1,
        "Kappa": kappa,
        "TrainTime_s": train_time,
        "TestTime_s": test_time,
        "InferenceTime_s": inference_time,
        "Memory_MB": end_mem - start_mem
    }])
    out_file = f"Performance_{tag}.xlsx"
    df_result.to_excel(out_file, index=False)
    print(f"  Saved results -> {out_file}")

    del ae, clf, enc_model
    gc.collect()
