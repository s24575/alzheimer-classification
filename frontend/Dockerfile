FROM python:3.12-slim-bookworm

RUN pip install poetry==2.0.1

ENV POETRY_NO_INTERACTION=true \
    POETRY_NO_ANSI=true \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN poetry install --with frontend --no-root

COPY frontend ./frontend

EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
