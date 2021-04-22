FROM python:3.7-slim
RUN apt update
RUN apt install -y git
ARG GDRIVE_ACCESS_KEY_ID
ARG GDRIVE_SECRET_ACCESS_KEY

WORKDIR /app
COPY . /app

RUN pip install dvc

RUN dvc remote modify myremote gdrive_client_id ${GDRIVE_ACCESS_KEY_ID}
RUN dvc remote modify myremote gdrive_client_secret ${GDRIVE_SECRET_ACCESS_KEY}
RUN pip install dvc[gdrive]
RUN dvc pull training_set

CMD ["python", "main.py"]