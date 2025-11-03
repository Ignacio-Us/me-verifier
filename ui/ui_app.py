import io
import json
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from pathlib import Path
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import torch
from dotenv import load_dotenv
import os
import time

# === Cargar variables de entorno ===
load_dotenv()

# === Configuración general ===
app = Flask(__name__, template_folder="templates", static_folder="static")

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.joblib"))
SCALER_PATH = MODEL_PATH.parent / "scaler.joblib"
DEFAULT_THRESHOLD = float(os.getenv("THRESHOLD", 0.75))
PORT = int(os.getenv("PORT", 5813))
MAX_FILE_SIZE_MB = float(os.getenv("MAX_MB", 5))
ALLOWED_EXTENSIONS = {"image/jpg", "image/jpeg", "image/png"}

# === Inicialización ===
print("[INFO] Cargando modelo y scaler...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

print("[INFO] Inicializando red FaceNet (InceptionResnetV1 + MTCNN)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = InceptionResnetV1(pretrained="vggface2").eval().to(device)
mtcnn = MTCNN(
    image_size=160,
    margin=20,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    keep_all=True,
    post_process=True,
    device=device
)

# === Configuración del umbral ===
threshold = DEFAULT_THRESHOLD
print(f"[INFO] Umbral configurado: τ={threshold:.4f}")


# === Funciones auxiliares ===
def validate_image(file):
    """Valida tipo MIME y tamaño del archivo."""
    if file.mimetype not in ALLOWED_EXTENSIONS:
        return False, "solo image/jpg, image/jpeg o image/png son permitidos"

    file.seek(0, os.SEEK_END)
    size_mb = file.tell() / (1024 * 1024)
    file.seek(0)

    if size_mb > MAX_FILE_SIZE_MB:
        return False, f"archivo demasiado grande ({size_mb:.2f} MB > {MAX_FILE_SIZE_MB} MB)"
    return True, None


def get_embedding(image: Image.Image):
    face_tensor = mtcnn(image)  # retorna tensor [N, 3, 160, 160]
    if face_tensor is None:
        raise ValueError("No se detectó ningún rostro.")

    with torch.no_grad():
        embeddings = embedder(face_tensor.to(device)).cpu().numpy()

    # si hay más de una cara, tomamos la primera
    return embeddings[0:1]



# === Ruta principal: interfaz web ===
@app.route("/", methods=["GET", "POST"])
def index():
    prediction, confidence = None, None
    timing_ms = None

    if request.method == "POST":
        start_time = time.time()
        try:
            file = request.files["file"]
            valid, error_msg = validate_image(file)
            if not valid:
                prediction = f"⚠️ {error_msg}"
            else:
                img = Image.open(file.stream).convert("RGB")
                embedding = get_embedding(img)
                embedding_scaled = scaler.transform(embedding)
                proba = model.predict_proba(embedding_scaled)[0, 1]

                is_me = proba >= threshold
                prediction = "✅ ME" if is_me else "❌ NOT ME"
                confidence = round(float(proba), 4)
                timing_ms = round((time.time() - start_time) * 1000, 1)

        except Exception as e:
            prediction = f"⚠️ Error: {e}"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        threshold=threshold,
        timing_ms=timing_ms
    )


# === Endpoint JSON (igual que API) ===
@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    try:
        file = request.files["image"]
        valid, error_msg = validate_image(file)
        if not valid:
            return jsonify({"error": error_msg}), 400

        img = Image.open(file.stream).convert("RGB")
        embedding = get_embedding(img)
        embedding_scaled = scaler.transform(embedding)

        prob = model.predict_proba(embedding_scaled)[0, 1]
        is_me = prob >= threshold

        return jsonify({
            "is_me": bool(is_me),
            "score": round(float(prob), 4),
            "threshold": round(float(threshold), 4),
            "timing_ms": round((time.time() - start_time) * 1000, 1)
        }), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"error": "error interno en el procesamiento"}), 500

# === Ejecutar aplicación ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
