from flask import Flask, render_template, request
import torch
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from PIL import Image
import joblib
import numpy as np
import os
from dotenv import load_dotenv

# === Cargar variables de entorno ===
load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.joblib")
THRESHOLD = float(os.getenv("THRESHOLD", 0.7))  # valor por defecto 0.7
PORT = int(os.getenv("PORT", 8500))

# === Configuración Flask ===
app = Flask(__name__, template_folder="templates", static_folder="static")
UPLOAD_FOLDER = "ui/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Cargar modelo y scaler ===
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# === Modelo de embeddings ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# === Transformaciones ===
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# === Función de predicción ===
def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = embedding_model(tensor).cpu().numpy().flatten()

    embedding_scaled = scaler.transform([embedding])
    probas = model.predict_proba(embedding_scaled)[0]
    print(probas)
    pred = np.argmax(probas)
    confidence = probas[pred]

    # Decisión basada en el threshold
    if confidence < THRESHOLD:
        label = "⚠️ Confianza baja"
    else:
        label = "✅ ME" if pred == 1 else "❌ NOT ME"

    return label, confidence

# === Rutas ===
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)
            prediction, confidence = predict_image(path)

    return render_template("index.html", prediction=prediction, confidence=confidence, threshold=THRESHOLD)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)
