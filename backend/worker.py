import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('worker')

def main():
    logger.error("❌ The old APScheduler background worker has been deprecated.")
    logger.info("✅ We have migrated to Celery for heavy background ML tasks!")
    logger.info("")
    logger.info("To run the background workers, please use the following commands in separate terminals:")
    logger.info("1. celery -A celery_app.celery worker --loglevel=info")
    logger.info("2. celery -A celery_app.celery beat --loglevel=info")
    logger.info("")
    logger.info("If you are using Docker, the docker-compose.yml has already been updated to run these for you.")
    sys.exit(1)

if __name__ == "__main__":
    main()
