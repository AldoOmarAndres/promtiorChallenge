FROM python:3.9
WORKDIR /deploy
COPY . /deploy
RUN pip install -r requirements.txt
CMD ["uvicorn", "deploy:app", "--host", "0.0.0.0", "--port", "8000"]