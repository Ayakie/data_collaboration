FROM python:3.7-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY poetry.lock pyproject.toml ./

RUN pip install poetry

RUN poetry config virtualenvs.create false \
  && poetry install 

RUN pip install jupyter_contrib_nbextensions && \
    jupyter contrib nbextension install --user && \
    jupyter nbextension enable highlight_selected_word/main &&\
    jupyter nbextension enable hinterland/hinterland
