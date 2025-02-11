FROM python:3.10-bookworm

WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt update -y
RUN apt install netcat-openbsd -y

# Install Poetry
RUN pip install poetry

# Copy poetry files
COPY pyproject.toml poetry.lock ./

# Configure poetry to not create virtual environment
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-root

# Copy project files
COPY . .

# Expose port
EXPOSE 8000

HEALTHCHECK CMD ["echo" "HEAD / HTTP/1.0" ">/dev/tcp/127.0.0.1/8000"]

# Run the application
CMD ["fastapi", "run", "src/app.py"]

