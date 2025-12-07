FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

#  ADD THIS
COPY serviceAccountKey.json /app/serviceAccountKey.json

COPY . /app
WORKDIR /app

EXPOSE 8000

CMD ["uvicorn", "gfsapi:app", "--host", "0.0.0.0", "--port", "8000"]
