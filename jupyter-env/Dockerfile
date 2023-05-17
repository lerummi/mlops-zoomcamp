FROM jupyter/datascience-notebook:python-3.10.8

LABEL maintainer="Martin Krause <martin.krause.19885@gmail.com>"

USER root

# Install apt packages
RUN apt-get update && \
    apt-get install -y git && \
    apt-get install --yes --no-install-recommends \
    && \
    pip install poetry

USER ${NB_UID}

# Requirements are installed here to ensure they will be cached.
COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock

# Install Python 3 packages
RUN pip install poetry && \
    poetry export -o /tmp/requirements.txt --without-hashes && \
    cat /tmp/requirements.txt

RUN pip install --use-deprecated=legacy-resolver  -r /tmp/requirements.txt
