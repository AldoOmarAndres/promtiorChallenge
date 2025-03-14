FROM python:3.10-slim

WORKDIR /deploy

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "deploy:app", "--host", "localhost", "--port", "8000"]