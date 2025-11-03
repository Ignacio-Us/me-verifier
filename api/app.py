import io
import json
import time
import joblib
import numpy as np
from flask import Flask, request, jsonify
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import torch
from pathlib import Path
from dotenv import load_dotenv
import os

# === Cargar variables de entorno ===
load_dotenv()

# === Configuración general ===
app = Flask(__name__)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.joblib"))
SCALER_PATH = MODEL_PATH.parent / "scaler.joblib"  # se asume en la misma carpeta
DEFAULT_THRESHOLD = float(os.getenv("THRESHOLD", 0.75))
PORT = int(os.getenv("PORT", 8000))
MAX_FILE_SIZE_MB = float(os.getenv("MAX_MB", 5))
ALLOWED_EXTENSIONS = {"image/jpg", "image/jpeg", "image/png"}

# === Cargar modelo, scaler y red de embeddings ===
print("[INFO] Cargando modelo y scaler...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

print("[INFO] Inicializando red InceptionResnetV1 + MTCNN...")
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = InceptionResnetV1(pretrained="vggface2").eval().to(device)
mtcnn = MTCNN(image_size=160, margin=20, device=device)

THRESHOLD = DEFAULT_THRESHOLD
print(f"[INFO] Umbral configurado: τ={THRESHOLD:.4f}")


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
    """Obtiene embedding facial 512-D de una imagen."""
    face = mtcnn(image)
    if face is None:
        raise ValueError("No se detectó ningún rostro.")
    with torch.no_grad():
        emb = embedder(face.unsqueeze(0).to(device)).cpu().numpy()
    return emb


# === Endpoint de salud ===
@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({
        "status": "ok",
        "device": device,
        "threshold": THRESHOLD,
        "model_path": str(MODEL_PATH)
    }), 200


# === Endpoint principal /verify ===
@app.route("/verify", methods=["POST"])
def verify():
    start_time = time.time()

    # Validar que se haya enviado un archivo
    if "image" not in request.files:
        return jsonify({"error": "no se envió ningún archivo"}), 400

    file = request.files["image"]

    # Validar tipo y tamaño
    valid, error_msg = validate_image(file)
    if not valid:
        return jsonify({"error": error_msg}), 400

    try:
        # Leer imagen
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Extraer embedding
        embedding = get_embedding(image)
        embedding_scaled = scaler.transform(embedding)

        # Calcular probabilidad de "me"
        prob = model.predict_proba(embedding_scaled)[0, 1]
        is_me = prob >= THRESHOLD

        # Tiempo total de inferencia
        timing_ms = (time.time() - start_time) * 1000

        return jsonify({
            "is_me": bool(is_me),
            "score": round(float(prob), 4),
            "threshold": round(float(THRESHOLD), 4),
            "timing_ms": round(float(timing_ms), 1)
        }), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"error": "error interno en el procesamiento"}), 500


# === Ejecutar aplicación localmente ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
