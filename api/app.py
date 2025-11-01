from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
import joblib
import numpy as np
from PIL import Image
from io import BytesIO
import base64

# === Cargar variables del archivo .env ===
load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "models/model_verifier.joblib")
THRESHOLD = float(os.getenv("THRESHOLD", 0.6))
PORT = int(os.getenv("PORT", 5813))
MAX_MB = int(os.getenv("MAX_MB", 8))

# === Inicializar Flask ===
app = Flask(__name__)

# === Cargar modelo ===
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Error cargando modelo desde {MODEL_PATH}: {e}")

# === Validadores ===
def validate_image(request):
    if "image" not in request.files:
        return False, "No se encontró archivo 'image' en la solicitud."

    file = request.files["image"]
    if file.filename == "":
        return False, "El archivo está vacío."

    file.seek(0, os.SEEK_END)
    file_size_mb = file.tell() / (1024 * 1024)
    file.seek(0)

    if file_size_mb > MAX_MB:
        return False, f"El archivo supera el límite de {MAX_MB} MB."

    return True, file

# === Endpoint de salud ===
@app.route("/health", methods=["GET"])
def healthz():
    return jsonify({"status": "ok", "model_loaded": MODEL_PATH}), 200

# === Endpoint principal ===
@app.route("/verify", methods=["POST"])
def verify():
    valid, result = validate_image(request)
    if not valid:
        return jsonify({"error": result}), 400

    file = result
    image = Image.open(file.stream).convert("RGB")

    # TODO: Aquí iría el cálculo de embeddings con facenet-pytorch
    # por ahora usaremos un vector simulado
    embedding = np.random.rand(512)  # placeholder

    # Clasificación o distancia con el modelo entrenado
    prediction = model.predict([embedding])[0]
    confidence = np.random.random()  # placeholder

    verified = confidence >= THRESHOLD

    return jsonify({
        "verified": bool(verified),
        "confidence": round(float(confidence), 4),
        "threshold": THRESHOLD
    }), 200

# === Ejecutar la app ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
