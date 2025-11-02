import requests

def test_healthz():
    r = requests.get("http://localhost:5000/healthz")
    print("OK")
    assert r.status_code == 200

def test_verify():
    with open("data/me/example.jpg", "rb") as img:
        r = requests.post("http://localhost:5000/verify", files={"image": img})
        print("OK")
        assert r.status_code == 200
        assert "verified" in r.json()
