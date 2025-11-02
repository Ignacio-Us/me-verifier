import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from pathlib import Path

# === Paths ===
EMBEDDINGS_PATH = Path("models/embeddings.joblib")
MODEL_PATH = Path("models/model_verifier.joblib")
SCALER_PATH = Path("models/scaler.joblib")

# === 1. Cargar embeddings ===
print("[INFO] Cargando embeddings...")
X, y = joblib.load(EMBEDDINGS_PATH)
print(f"[INFO] Datos cargados: X={X.shape}, y={y.shape}")

# === 2. Dividir conjunto de entrenamiento / validación ===
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === 3. Escalado de datos ===
print("[INFO] Escalando embeddings...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# === 4. Inicializar y entrenar modelo ===
print("[INFO] Entrenando modelo...")
model = LogisticRegression(max_iter=300, random_state=42, class_weight="balanced")
model.fit(X_train_scaled, y_train)

print("[OK] Modelo entrenado correctamente")

# === 5. Guardar modelo y scaler ===
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print(f"[OK] Modelo guardado en {MODEL_PATH}")
print(f"[OK] Scaler guardado en {SCALER_PATH}")

# === 6. Guardar conjunto de validación para evaluación futura ===
VAL_PATH = Path("models/val_data.joblib")
joblib.dump((X_val, y_val), VAL_PATH)
print(f"[OK] Conjunto de validación guardado en {VAL_PATH}")