import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)

# === Paths ===
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "model.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
EMBEDDINGS_PATH = MODELS_DIR / "embeddings.npy"
LABELS_PATH = MODELS_DIR / "labels.csv"

# === Cargar modelo, scaler y datos ===
print("[INFO] Cargando modelo, scaler y embeddings...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
X = np.load(EMBEDDINGS_PATH)
labels_df = pd.read_csv(LABELS_PATH)
y_true = labels_df["label"].values

# === Escalar embeddings ===
print("[INFO] Escalando embeddings...")
X_scaled = scaler.transform(X)

# === Predicciones ===
print("[INFO] Calculando predicciones y probabilidades...")
y_proba = model.predict_proba(X_scaled)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)  # umbral inicial = 0.5

# === Métricas principales ===
roc_auc = roc_auc_score(y_true, y_proba)
pr_auc = average_precision_score(y_true, y_proba)
cm = confusion_matrix(y_true, y_pred)

# === Umbral óptimo (Youden's J statistic) ===
fpr, tpr, thresholds = roc_curve(y_true, y_proba)
j_scores = tpr - fpr
j_best_idx = np.argmax(j_scores)
best_thresh = thresholds[j_best_idx]
print(f"[INFO] Umbral óptimo (τ): {best_thresh:.4f}")

# === Guardar métricas ===
metrics = {
    "roc_auc": round(float(roc_auc), 4),
    "pr_auc": round(float(pr_auc), 4),
    "best_threshold": round(float(best_thresh), 4),
    "confusion_matrix": cm.tolist(),
}

with open(REPORTS_DIR / "metrics_eval.json", "w") as f:
    json.dump(metrics, f, indent=4)

# === Graficar Matriz de Confusión ===
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["not_me", "me"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Matriz de Confusión - Verificador Facial")
plt.savefig(REPORTS_DIR / "confusion_matrix.png", bbox_inches="tight")
plt.close()

# === Curva ROC ===
plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC")
plt.legend()
plt.grid(True)
plt.savefig(REPORTS_DIR / "roc_curve.png", bbox_inches="tight")
plt.close()

# === Curva Precision–Recall ===
precision, recall, _ = precision_recall_curve(y_true, y_proba)
plt.figure()
plt.plot(recall, precision, label=f"PR curve (AP = {pr_auc:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Curva Precision–Recall")
plt.legend()
plt.grid(True)
plt.savefig(REPORTS_DIR / "pr_curve.png", bbox_inches="tight")
plt.close()

# === Resumen ===
print("\n=== RESULTADOS ===")
for k, v in metrics.items():
    print(f"{k}: {v}")

print(f"\n[OK] Reportes guardados en {REPORTS_DIR.resolve()}")
