import os
import json
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from pathlib import Path
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import torch

# === Configuración inicial ===
app = Flask(__name__, template_folder="templates", static_folder="static")

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
DEFAULT_THRESHOLD = float(os.getenv("THRESHOLD", 0.5))

# === Cargar modelo, scaler y FaceNet ===
print("[INFO] Cargando modelo y scaler...")
model = joblib.load(MODELS_DIR / "model.joblib")
scaler = joblib.load(MODELS_DIR / "scaler.joblib")

print("[INFO] Inicializando modelo de embeddings (InceptionResnetV1)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = InceptionResnetV1(pretrained="vggface2").eval().to(device)
mtcnn = MTCNN(
    image_size=160,
    margin=20,
    thresholds=[0.6, 0.7, 0.7],  # menor → más sensible, más falsos positivos
    factor=0.709,
    keep_all=True,               # detectar todas las caras
    post_process=True,
    device=device
)

# === Leer umbral óptimo si existe ===
metrics_path = REPORTS_DIR / "metrics_eval.json"
if metrics_path.exists():
    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
            threshold = float(metrics.get("best_threshold", DEFAULT_THRESHOLD))
        print(f"[INFO] Umbral óptimo (τ) cargado: {threshold:.4f}")
    except Exception as e:
        print(f"[WARN] No se pudo leer metrics_eval.json: {e}")
        threshold = DEFAULT_THRESHOLD
else:
    print("[WARN] metrics_eval.json no encontrado. Usando umbral por defecto.")
    threshold = DEFAULT_THRESHOLD


# === Función para obtener embedding de imagen ===
def get_embedding(image: Image.Image):
    # Detectar todas las caras
    boxes, probs = mtcnn.detect(image)
    if boxes is None or len(boxes) == 0:
        raise ValueError("No se detectó ningún rostro.")

    # Seleccionar la cara con mayor probabilidad
    best_idx = int(np.argmax(probs))
    box = boxes[best_idx].astype(int)

    # Recortar manualmente
    face = image.crop(box)

    # Transformar a tensor con el mismo formato que MTCNN
    face_tensor = mtcnn.prewhiten(np.array(face))
    face_tensor = torch.tensor(face_tensor).permute(2, 0, 1).unsqueeze(0).float().to(device)

    # Obtener embedding
    with torch.no_grad():
        embedding = embedder(face_tensor).cpu().numpy()

    return embedding


# === Ruta principal: interfaz web ===
@app.route("/", methods=["GET", "POST"])
def index():
    prediction, confidence = None, None

    if request.method == "POST":
        try:
            file = request.files["file"]
            img = Image.open(file.stream).convert("RGB")

            # Embedding + predicción
            embedding = get_embedding(img)
            embedding_scaled = scaler.transform(embedding)
            proba = model.predict_proba(embedding_scaled)[0, 1]

            # Clasificación
            is_me = proba >= threshold
            prediction = "✅ ME" if is_me else "❌ NOT ME"
            confidence = proba

        except Exception as e:
            prediction = f"⚠️ Error: {e}"

    return render_template("index.html", prediction=prediction, confidence=confidence)


# === Endpoint JSON separado (opcional, para API directa) ===
@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["image"]
        img = Image.open(file.stream).convert("RGB")

        embedding = get_embedding(img)
        embedding_scaled = scaler.transform(embedding)
        proba = model.predict_proba(embedding_scaled)[0, 1]

        is_me = proba >= threshold
        result = {
            "probability": round(float(proba), 4),
            "threshold": round(float(threshold), 4),
            "prediction": "me" if is_me else "not_me",
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# === Inicio del servidor ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5813)))
