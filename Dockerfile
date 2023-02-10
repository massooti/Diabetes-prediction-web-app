FROM python:3.10-slim

WORKDIR /app

# RUN apt-get update && apt-get install -y \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 8501



CMD [ "streamlit", "run","app.py" ]