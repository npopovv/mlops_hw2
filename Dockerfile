# Указываем базовый образ с Python 3.12
FROM python:3.12.3-slim

# Устанавливаем зависимости системы
# git для dvc
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl-dev \
    libffi-dev \
    git \
    && rm -rf /var/lib/apt/lists/*


# Настройка Git
RUN git config --global user.name "nkk" \
    && git config --global user.email "nkk@gmail.com"

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем pyproject.toml и poetry.lock для установки зависимостей
COPY pyproject.toml poetry.lock /app/

# Устанавливаем Poetry
RUN pip install --no-cache-dir poetry

# Устанавливаем зависимости через Poetry
RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi

# Копируем всё приложение
COPY . /app

# Указываем порт, на котором будет работать приложение
EXPOSE 8000

# Команда для запуска приложения
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]