import numpy as np

from oc_sentiment_ml_api.model import predict


def test_predict_function_output():
    label, proba = predict("J'aime beaucoup ce produit")
    assert isinstance(label, str)
    assert isinstance(proba, np.float32)
    assert label in ["positive", "negative"]
    assert 0 <= proba <= 1.0
