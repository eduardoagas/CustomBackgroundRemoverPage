FROM python:3.10

## App engine stuff

# Upgrade pip 
RUN pip install -U pip

COPY requirements.txt app/requirements.txt
RUN pip install -r app/requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY websocket.py /usr/local/lib/python3.10/site-packages/tornado


# Create a new directory for app (keep it in its own directory)
COPY . /app
WORKDIR /app

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]