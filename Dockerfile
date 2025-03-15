#Dejo el archivo de docker solo para mantener registro de lo que fui haciendo y las alternativas
# que fui probando

FROM python:3.10-slim

WORKDIR /deploy

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "deploy:app", "--host", "0.0.0.0", "--port", "8000"]