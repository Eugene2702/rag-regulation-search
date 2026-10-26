FROM python:3.11-slim

WORKDIR /app

# Копируем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY ./app ./app
COPY evaluate.py .
COPY testset.json .

# Открываем порт
EXPOSE 8000

# Запускаем сервер (без --reload, так как в докере это не нужно)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]