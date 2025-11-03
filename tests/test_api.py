# tests/test_api_verify.py
import pytest
import requests
import os
from dotenv import load_dotenv

# === Cargar variables de entorno ===
load_dotenv()
BASE_URL = f"http://localhost:{os.getenv('PORT', 5813)}"


# === Fixtures reutilizables ===
@pytest.fixture(scope="module")
def base_url():
    """Fixture para reutilizar la URL base de la API."""
    return BASE_URL


@pytest.fixture
def image_me_path():
    path = "data/test/me/image_test_api.jpeg"
    assert os.path.exists(path), f"Imagen no encontrada: {path}"
    return path


@pytest.fixture
def image_not_me_path():
    path = "data/test/not_me/001383.jpg"
    assert os.path.exists(path), f"Imagen no encontrada: {path}"
    return path


# === TESTS ===

def test_healthz(base_url):
    """Verifica que el endpoint /healthz responda correctamente."""
    r = requests.get(f"{base_url}/healthz")
    assert r.status_code == 200, f"Error en /healthz: {r.status_code} {r.text}"

    data = r.json()
    assert data.get("status") == "ok", "Campo 'status' incorrecto"
    assert "threshold" in data, "Campo 'threshold' faltante"
    assert "model_path" in data, "Campo 'model_path' faltante"


def test_verify_me(base_url, image_me_path):
    """Verifica el endpoint /verify con una imagen tipo 'me'."""
    with open(image_me_path, "rb") as img:
        files = {"image": ("image_test_api.jpeg", img, "image/jpeg")}
        r = requests.post(f"{base_url}/verify", files=files)

    assert r.status_code == 200, f"Error en /verify (me): {r.status_code} {r.text}"
    data = r.json()

    assert "is_me" in data, "Campo 'is_me' faltante"
    assert "score" in data, "Campo 'score' faltante"


def test_verify_not_me(base_url, image_not_me_path):
    """Verifica el endpoint /verify con una imagen tipo 'not_me'."""
    with open(image_not_me_path, "rb") as img:
        files = {"image": ("001383.jpg", img, "image/jpeg")}
        r = requests.post(f"{base_url}/verify", files=files)

    assert r.status_code == 200, f"Error en /verify (not_me): {r.status_code} {r.text}"
    data = r.json()

    assert "is_me" in data, "Campo 'is_me' faltante"
    assert "score" in data, "Campo 'score' faltante"


def test_img_format_invalid(base_url, image_me_path):
    """Verifica el endpoint /verify con una imagen de formato inválido."""
    with open(image_me_path, "rb") as img:
        # Enviar con tipo MIME inválido
        files = {"image": ("image_test_api.jpeg", img, "text/plain")}
        r = requests.post(f"{base_url}/verify", files=files)

    assert r.status_code == 400, f"Se esperaba 400 pero se recibió {r.status_code}"
    data = r.json()
    assert "error" in data, "Campo 'error' faltante en la respuesta"
