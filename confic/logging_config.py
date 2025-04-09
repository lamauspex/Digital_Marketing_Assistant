
from logging.handlers import RotatingFileHandler
import logging


# Конфигурация логирования с ротацией логов
logger = logging.getLogger('NeuroAssistantLogger')
logger.setLevel(logging.INFO)
handler = RotatingFileHandler('neuro_assistant.log', maxBytes=5*1024*1024, backupCount=2)  # 5 МБ на файл, 2 резервные копии
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)