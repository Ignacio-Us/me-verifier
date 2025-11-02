from pathlib import Path
import joblib
import numpy as np
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# === Paths ===
MODEL_PATH = Path("models/model_verifier.joblib")
SCALER_PATH = Path("models/scaler.joblib")
VAL_PATH = Path("models/val_data.joblib")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

# === Cargar modelo, escalador y conjunto de validación ===
print("[INFO] Cargando modelo, escalador y conjunto de validación...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
X_val, y_val = joblib.load(VAL_PATH)

# === Escalar embeddings ===
print("[INFO] Escalando embeddings de validación...")
X_val_scaled = scaler.transform(X_val)

# === Predicciones ===
print("[INFO] Realizando predicciones...")
y_pred = model.predict(X_val_scaled)

# === Calcular métricas ===
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
cm = confusion_matrix(y_val, y_pred)

# === Mostrar y guardar métricas ===
metrics = {
    "accuracy": round(accuracy, 4),
    "precision": round(precision, 4),
    "recall": round(recall, 4),
    "f1_score": round(f1, 4)
}

print("\n=== 📊 Métricas de Evaluación ===")
for k, v in metrics.items():
    print(f"{k}: {v}")

with open(REPORTS_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["not_me", "me"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Matriz de Confusión - Validación del Verificador")
plt.savefig(REPORTS_DIR / "confusion_matrix.png", bbox_inches="tight")
plt.close()

print(f"\n[OK] Métricas guardadas en {REPORTS_DIR / 'metrics.json'}")
print(f"[OK] Matriz de confusión guardada en {REPORTS_DIR / 'confusion_matrix.png'}")