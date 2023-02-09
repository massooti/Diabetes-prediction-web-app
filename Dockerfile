# app/Dockerfile

FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt ./requirements.txt

# RUN apt-get update && apt-get install -y \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

RUN pip3 install -r requirements.txt

EXPOSE 8501

COPY . /app

ENTRYPOINT ["streamlit", "run"]

CMD [ "app.py" ]