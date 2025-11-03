import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import json

# === Paths ===
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
EMBEDDINGS_PATH = MODELS_DIR / "embeddings.npy"
LABELS_PATH = MODELS_DIR / "labels.csv"
MODEL_PATH = MODELS_DIR / "model.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# === Cargar datos ===
print("[INFO] Cargando embeddings y etiquetas...")
X = np.load(EMBEDDINGS_PATH)
labels_df = pd.read_csv(LABELS_PATH)
y = labels_df["label"].values

print(f"[INFO] Datos cargados: X={X.shape}, y={y.shape}")

# === Dividir train / validation ===
print("[INFO] Dividiendo en conjuntos de entrenamiento y validación...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Escalar datos ===
print("[INFO] Escalando datos...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# === Entrenar modelo ===
print("[INFO] Entrenando modelo LogisticRegression...")
model = LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced")
model.fit(X_train_scaled, y_train)
print("[OK] Modelo entrenado correctamente")

# === Evaluar ===
print("[INFO] Evaluando modelo...")
y_pred = model.predict(X_val_scaled)
y_proba = model.predict_proba(X_val_scaled)[:, 1]

accuracy = accuracy_score(y_val, y_pred)
auc = roc_auc_score(y_val, y_proba)

metrics = {
    "accuracy": round(float(accuracy), 4),
    "auc": round(float(auc), 4),
}

# === Guardar artefactos ===
print("[INFO] Guardando modelo y escalador...")
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("[INFO] Guardando métricas...")
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=4)

# === Resumen ===
print("\n=== RESULTADOS ===")
for k, v in metrics.items():
    print(f"{k}: {v}")

print(f"\n[OK] Modelo guardado en: {MODEL_PATH}")
print(f"[OK] Scaler guardado en: {SCALER_PATH}")
print(f"[OK] Métricas guardadas en: {METRICS_PATH}")