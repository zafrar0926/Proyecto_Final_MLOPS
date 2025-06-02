FROM apache/airflow:2.7.1

USER root
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

USER airflow
COPY requirements.txt /requirements.txt
RUN pip install --user -r /requirements.txt

ENV PYTHONPATH=/home/airflow/.local/lib/python3.8/site-packages:$PYTHONPATH 