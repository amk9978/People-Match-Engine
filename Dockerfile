FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "presentation.api_controller:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]