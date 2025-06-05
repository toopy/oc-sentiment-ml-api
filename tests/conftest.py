import pytest
from fastapi.testclient import TestClient

from oc_sentiment_ml_api.main import app


@pytest.fixture
def client():
    return TestClient(app)
