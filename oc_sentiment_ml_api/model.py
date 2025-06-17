import joblib
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

from .utils import clean_text

model = joblib.load("model/model.pkl")
tokenizer = joblib.load("model/tokenizer.pkl")

MAX_LEN = 100


def predict(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    proba = model.predict(padded)[0]
    label = "positive" if int(np.argmax(proba)) else "negative"
    return label, max(proba)
