FROM python:3.10-bookworm

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt update -y
RUN apt install netcat-openbsd -y

# Copy poetry files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv pip install --system -r pyproject.toml

# Copy project files
COPY . .

# Expose port
EXPOSE 8000

HEALTHCHECK CMD curl -f http://127.0.0.1:8000 || exit 1

# Run the application
CMD ["python", "src/app.py"]

