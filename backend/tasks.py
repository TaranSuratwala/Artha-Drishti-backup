import logging
from celery_app import celery
from application import run_feature_engineering, run_sentiment_precache, predictor

logger = logging.getLogger(__name__)

@celery.task(bind=True, max_retries=3)
def run_feature_engineering_task(self):
    """
    Celery task to run feature engineering in the background.
    """
    try:
        logger.info("Starting feature engineering celery task...")
        run_feature_engineering()
        logger.info("Feature engineering celery task completed successfully.")
        return "Success"
    except Exception as exc:
        logger.error(f"Feature engineering task failed: {exc}")
        raise self.retry(exc=exc, countdown=60)


@celery.task(bind=True, max_retries=3)
def run_sentiment_precache_task(self):
    """
    Celery task to pre-cache sentiment analysis.
    """
    try:
        logger.info("Starting sentiment pre-cache celery task...")
        run_sentiment_precache()
        logger.info("Sentiment pre-cache celery task completed successfully.")
        return "Success"
    except Exception as exc:
        logger.error(f"Sentiment pre-cache task failed: {exc}")
        raise self.retry(exc=exc, countdown=60)


@celery.task(bind=True, max_retries=3)
def train_model_task(self, ticker):
    """
    Celery task to train the model for a specific ticker.
    """
    try:
        logger.info(f"Starting model training celery task for {ticker}...")
        result = predictor.train(ticker)
        logger.info(f"Model training celery task for {ticker} completed.")
        return result
    except Exception as exc:
        logger.error(f"Model training task failed for {ticker}: {exc}")
        raise self.retry(exc=exc, countdown=60)

@celery.task(bind=True, max_retries=1)
def train_unified_model_task(self):
    """
    Celery task to train the unified model for all tickers (weekly fine-tuning).
    """
    try:
        logger.info("Starting weekly unified model training celery task...")
        result = predictor.train(max_tickers=None)
        logger.info("Weekly unified model training celery task completed.")
        return "Success"
    except Exception as exc:
        logger.error(f"Unified model training task failed: {exc}")
        raise self.retry(exc=exc, countdown=300)

@celery.task(bind=True, time_limit=86400)
def certify_model_task(self, k_seeds=5):
    """
    Celery task to run multi-seed certification.
    """
    try:
        from multi_seed_certifier import MultiSeedCertifier
        from MLPredictor import UnifiedStockPredictor
        certifier = MultiSeedCertifier(UnifiedStockPredictor, k_seeds=k_seeds)
        report = certifier.certify()
        return {"ready": report.deployment_ready, "report_path": report.timestamp}
    except Exception as exc:
        logger.error(f"Certification task failed: {exc}")
        raise self.retry(exc=exc, countdown=300)
