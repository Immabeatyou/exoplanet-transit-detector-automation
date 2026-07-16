import os
from datetime import timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

class Config:
    os.makedirs(DATA_DIR, exist_ok=True)
    # For production on Render, use a simple SQLite path; for local dev, use data/ folder
    if os.environ.get('FLASK_ENV') == 'production':
        SQLALCHEMY_DATABASE_URI = 'sqlite:////tmp/transit_app.db'
    else:
        SQLALCHEMY_DATABASE_URI = os.environ.get(
            'DATABASE_URL',
            f'sqlite:///{os.path.join(DATA_DIR, "transit_app.db")}'
        )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    UPLOAD_FOLDER = os.path.join(DATA_DIR, 'uploads')
    RESULTS_FOLDER = os.path.join(DATA_DIR, 'results')
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024 
class DevelopmentConfig(Config):
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    DEBUG = False
    TESTING = False

class TestingConfig(Config):
    DEBUG = True
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}