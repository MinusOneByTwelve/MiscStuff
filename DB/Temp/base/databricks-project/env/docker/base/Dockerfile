FROM jupyter/pyspark-notebook
COPY requirements.txt .
RUN pip install -r requirements.txt
ENV PYTHONPATH ${PYTHONPATH}:/home/jovyan/src
