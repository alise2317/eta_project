FROM python:3.11-slim

WORKDIR /code

# Ставим зависимости
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Копируем всё остальное (код app, папку ml и модель если она в корне)
COPY . /code

# Запуск
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
