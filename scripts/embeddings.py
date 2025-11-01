import torch
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from torchvision import transforms
from pathlib import Path
import joblib
import numpy as np

DATA_DIR = Path("data/cropped")
OUTPUT_PATH = Path("models/embeddings.joblib")

# Transformaciones para InceptionResnetV1
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Cargar modelo preentrenado de FaceNet (VGGFace2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

X, y = [], []

# Recorremos las carpetas me/ y not_me/
for class_dir in ["me", "not_me"]:
    label = 1 if class_dir == "me" else 0
    image_dir = DATA_DIR / class_dir

    for img_path in image_dir.glob("*.jpg"):
        try:
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                embedding = model(tensor).cpu().numpy().flatten()

            X.append(embedding)
            y.append(label)
            print(f"[OK] {img_path.name} -> embedding generado")

        except Exception as e:
            print(f"[ERROR] {img_path.name}: {e}")

# Guardamos embeddings y etiquetas
X, y = np.array(X), np.array(y)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump((X, y), OUTPUT_PATH)

print(f"\n✅ Embeddings guardados en {OUTPUT_PATH}")
print(f"Total muestras: {len(X)}")
