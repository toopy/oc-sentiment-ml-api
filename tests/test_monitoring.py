from unittest.mock import patch

from oc_sentiment_ml_api.monitoring import log_bad_prediction


@patch("oc_sentiment_ml_api.monitoring.logger")
def test_log_bad_prediction(mock_logger):
    log_bad_prediction("[pytest] It was a good movie", "positive")
    mock_logger.warning.assert_called_once()
