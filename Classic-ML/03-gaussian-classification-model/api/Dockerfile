# Використовуємо офіційний Python образ
FROM python:3.11-slim

# Встановлюємо робочу директорію
WORKDIR /app

# Копіюємо файли залежностей
COPY requirements.txt .

# Встановлюємо залежності
RUN pip install --no-cache-dir -r requirements.txt

# Копіюємо код додатку
COPY main.py .

# Створюємо директорію для даних
RUN mkdir -p /app/datasets

# Відкриваємо порт
EXPOSE 8002

# Команда запуску
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]

