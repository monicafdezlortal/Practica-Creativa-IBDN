FROM python:3.7-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt


EXPOSE 5000
CMD ["python3", "predict_flask.py"]

