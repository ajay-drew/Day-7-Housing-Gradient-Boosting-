FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY model.py .
COPY train.csv .
COPY test.csv .
VOLUME /app/
ENV PYTHONUNBUFFERED=1
CMD ["python", "model.py"]