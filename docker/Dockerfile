FROM python:3.10-bookworm

WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt update -y
RUN apt install netcat-openbsd -y

# Install Poetry
RUN pip install uv

# Copy poetry files
COPY pyproject.toml uv.lock ./

# Configure poetry to not create virtual environment
# RUN poetry config virtualenvs.create false

# Install dependencies
RUN uv sync

# Copy project files
COPY . .

# Expose port
EXPOSE 8000

HEALTHCHECK CMD curl -f http://127.0.0.1:8000 || exit 1

# Run the application
CMD ["python", "src/app.py"]

