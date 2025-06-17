# Base image
FROM mcr.microsoft.com/azureml/minimal-py312-inference

# Working directory
WORKDIR /app

# Copy requirements and install
COPY setup.* .
COPY setup.* .
COPY pyproject.toml .
RUN mkdir -p ./oc_sentiment_ml_api \
 && touch ./oc_sentiment_ml_api/__init__.py \
 && pip install --no-cache-dir -e .

# Copy source code
COPY . .

# Port pour Azure / Render
ENV PORT=8000

# Expose the port
EXPOSE 8000

# Commande de lancement
CMD ["uvicorn", "oc_sentiment_ml_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
