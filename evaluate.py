import joblib
import numpy as np
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from pathlib import Path

# === Paths ===
MODEL_PATH = Path("models/model.joblib")
SCALER_PATH = Path("models/scaler.joblib")
EMBEDDINGS_PATH = Path("models/embeddings.joblib")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

# === Cargar modelo, escalador y embeddings ===
print("[INFO] Cargando modelo, escalador y embeddings...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
X, y = joblib.load(EMBEDDINGS_PATH)

# === Escalar embeddings ===
print("[INFO] Escalando embeddings...")
X_scaled = scaler.transform(X)

# === Predicciones ===
print("[INFO] Realizando predicciones...")
y_pred = model.predict(X_scaled)

# === Calcular métricas ===
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
cm = confusion_matrix(y, y_pred)

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

# === Guardar resultados ===
with open(REPORTS_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
print(f"\n[OK] Métricas guardadas en {REPORTS_DIR / 'metrics.json'}")

# === Graficar matriz de confusión ===
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["not_me", "me"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Matriz de Confusión - Verificador de Identidad")
plt.savefig(REPORTS_DIR / "confusion_matrix.png", bbox_inches="tight")
plt.close()
print(f"[OK] Matriz de confusión guardada en {REPORTS_DIR / 'confusion_matrix.png'}")
