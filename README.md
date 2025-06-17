# My OpenClassrooms Sentiment Model API

A RESTful API for sentiment analysis built with FastAPI. It predicts the sentiment (positive/negative) of a given text.

## Features

- Predict sentiment from text input
- FastAPI-based web server
- Easy to deploy with Uvicorn
- Ready for Docker deployment

## Install

```bash
pip install -e .
```

## Linting

```bash
pip install -e '.[lint]'
pre-commit run --all-files
```

## Test

```bash
pip install -e '.[test]'
pytest -sxv tests/
```

## Run

```bash
uvicorn oc_sentiment_ml_api.main:app --host 0.0.0.0 --port 8000
```

## Example Usage

```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"text": "I love this product!"}'
```

Response:
```json
{
  "sentiment": "positive",
  "confidence": 0.98
}
```

## Contributing

Pull requests are welcome! Please open an issue first to discuss changes.

## License

MIT License
