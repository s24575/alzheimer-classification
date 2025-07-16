FROM python:3.12-slim-bookworm

RUN pip install poetry==2.0.1

ENV POETRY_NO_INTERACTION=true \
    POETRY_NO_ANSI=true \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN poetry install --with backend --no-root

COPY backend ./backend
COPY alzheimer_classification ./alzheimer_classification

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
