import os
from celery import Celery
from celery.schedules import crontab
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="eventlet")
warnings.filterwarnings("ignore", message=".*Eventlet is deprecated.*")

load_dotenv()

redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

celery = Celery(
    'arthadrishti_tasks',
    broker=redis_url,
    backend=redis_url,
    include=['tasks']
)

celery.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Asia/Kolkata',
    enable_utc=True,
    # Configure beat schedule
    beat_schedule={
        'daily_feature_engineering': {
            'task': 'tasks.run_feature_engineering_task',
            'schedule': crontab(hour=16, minute=0, day_of_week='mon-fri'), # 4 PM IST Mon-Fri
        },
        'daily_sentiment_precache': {
            'task': 'tasks.run_sentiment_precache_task',
            'schedule': crontab(hour=16, minute=30, day_of_week='mon-fri'), # 4:30 PM IST Mon-Fri
        },
        'weekly_model_training': {
            'task': 'tasks.train_unified_model_task',
            'schedule': crontab(hour=2, minute=0, day_of_week='sat'), # 2:00 AM IST Saturday
        }
    }
)
