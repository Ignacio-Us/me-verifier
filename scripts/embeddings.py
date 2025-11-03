import torch
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from torchvision import transforms
from pathlib import Path
import numpy as np
import pandas as pd

# === Paths ===
DATA_DIR = Path("data/cropped")
OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDINGS_PATH = OUTPUT_DIR / "embeddings.npy"
LABELS_CSV_PATH = OUTPUT_DIR / "labels.csv"

# === Transformaciones para el modelo ===
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# === Modelo FaceNet (VGGFace2 preentrenado) ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# === Contenedores ===
embeddings = []
records = []

# === Recorremos las carpetas 'me' y 'not_me' ===
for class_dir in ["me", "not_me"]:
    label = 1 if class_dir == "me" else 0
    img_dir = DATA_DIR / class_dir

    if not img_dir.exists():
        print(f"[WARN] Directorio no encontrado: {img_dir}")
        continue

    for img_path in img_dir.glob("*.jpg"):
        try:
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                embedding = model(tensor).cpu().numpy().flatten()

            embeddings.append(embedding)
            records.append({
                "filename": img_path.name,
                "label": label
            })
            print(f"[OK] {img_path.name} -> embedding generado")

        except Exception as e:
            print(f"[ERROR] {img_path.name}: {e}")

    for img_path in img_dir.glob("*.jpeg"):
        try:
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                embedding = model(tensor).cpu().numpy().flatten()

            embeddings.append(embedding)
            records.append({
                "filename": img_path.name,
                "label": label
            })
            print(f"[OK] {img_path.name} -> embedding generado")

        except Exception as e:
            print(f"[ERROR] {img_path.name}: {e}")

# === Convertir y guardar ===
embeddings = np.array(embeddings, dtype=np.float32)
labels_df = pd.DataFrame(records)

np.save(EMBEDDINGS_PATH, embeddings)
labels_df.to_csv(LABELS_CSV_PATH, index=False)

print("\n[OK] Embeddings y etiquetas guardadas correctamente")
print(f"Embeddings: {EMBEDDINGS_PATH} (shape={embeddings.shape})")
print(f"Etiquetas:  {LABELS_CSV_PATH} (total={len(labels_df)})")
