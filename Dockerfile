FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev

COPY main.py ./
RUN mkdir -p /app/tmp

EXPOSE 8080

CMD ["sh", "-c", "exec uv run --no-sync uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]