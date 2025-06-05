from fastapi.testclient import TestClient

from oc_sentiment_ml_api.main import app

client = TestClient(app)


def test_feedback_wrong_prediction(client):
    data = {"text": "[pytest] It was a good movie", "prediction": "positive"}
    response = client.post("/feedback", json=data)
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_valid_input(client):
    response = client.post("/predict", json={"text": "I love this product !"})
    assert response.status_code == 200
    assert "confidence" in response.json()
    assert "sentiment" in response.json()


def test_predict_empty_input(client):
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 200  # ou 422 si tu valides côté Pydantic
    assert "confidence" in response.json()
    assert "sentiment" in response.json()


def test_predict_special_characters(client):
    response = client.post("/predict", json={"text": "😡💥!! it's bad"})
    assert response.status_code == 200
    assert isinstance(response.json()["confidence"], float)
    assert isinstance(response.json()["sentiment"], str)


def test_predict_missing_field(client):
    response = client.post("/predict", json={})
    assert response.status_code == 422
