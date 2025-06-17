import logging
import os

from opencensus.ext.azure.log_exporter import AzureLogHandler

logger = logging.getLogger("sentiment_monitoring")

APPINSIGHTS_CONNECTION_STRING = os.getenv("APPINSIGHTS_CONNECTION_STRING")

if APPINSIGHTS_CONNECTION_STRING:
    logger.addHandler(AzureLogHandler(connection_string=APPINSIGHTS_CONNECTION_STRING))

logger.setLevel(logging.INFO)


def log_bad_prediction(text, prediction):
    logger.warning(
        "Bad prediction",
        extra={
            "custom_dimensions": {
                "tweet_text": text,
                "predicted_label": prediction,
            }
        },
    )
