import logging
from datetime import datetime


def getFileName():
    now = datetime.now()
    formatted_day = now.strftime("%d-%m-%Y-%H-%M-%S")

    return formatted_day


class Logger:
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(f'logs/{getFileName()}.log')
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    @classmethod
    def getLogger(cls, name):
        logger = cls.logger
        logger.name = name
        return logger
