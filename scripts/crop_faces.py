import os
from PIL import Image, ImageOps
from facenet_pytorch import MTCNN
from pathlib import Path
import numpy as np

# Directorios de entrada/salida
INPUT_DIRS = ["data/me", "data/not_me"]
OUTPUT_DIR = Path("data/cropped")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Inicializa detector MTCNN
mtcnn = MTCNN(image_size=160, margin=20)

for input_dir in INPUT_DIRS:
    label = Path(input_dir).name
    output_subdir = OUTPUT_DIR / label
    output_subdir.mkdir(parents=True, exist_ok=True)

    for img_name in os.listdir(input_dir):
        img_path = Path(input_dir) / img_name
        try:
            img = Image.open(img_path).convert("RGB")
            img = ImageOps.exif_transpose(img)  # Corrige orientación antes de pasar al detector
            face = mtcnn(img)

            if face is not None:
                # Normaliza valores dinámicamente para evitar imágenes oscuras
                face_np = face.permute(1, 2, 0).cpu().numpy()
                face_np = ((face_np - face_np.min()) / (face_np.max() - face_np.min()) * 255).astype(np.uint8)
                face_img = Image.fromarray(face_np)

                out_path = output_subdir / img_name
                face_img.save(out_path)
                print(f"[OK] {img_name} -> {out_path}")
            else:
                print(f"[WARN] No face detected in {img_name}")
        except Exception as e:
            print(f"[ERROR] {img_name}: {e}")
