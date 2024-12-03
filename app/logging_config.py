import logging


def setup_logger(
    name: str = None, log_file: str = "app.log", level: int = logging.INFO
):
    """
    Настраивает логгер с указанным именем, файлом и уровнем логирования.
    :param name: Имя логгера (по умолчанию None для root-логгера).
    :param log_file: Имя файла для записи логов.
    :param level: Уровень логирования (например, logging.INFO, logging.DEBUG).
    :return: Настроенный объект логгера.
    """
    # создаём форматтер
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # чоздаём обработчик для файла
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # чоздаём обработчик для консоли
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # инициализация логгера
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # добавляем обработчики
    if not logger.handlers:  # чтобы не дублировать обработчики при повторной настройке
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
