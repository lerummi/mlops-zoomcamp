FROM  svizor/zoomcamp-model:mlops-3.10.0-slim

RUN apt-get update && \
    apt-get clean && \
    pip install --upgrade pip && \
    pip install pipenv

RUN pipenv install scikit-learn==1.2.2 pandas pyarrow


WORKDIR /app
COPY starter.py .

ENTRYPOINT []