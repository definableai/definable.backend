#!/bin/bash

# Shell script to start Celery worker for KB service

echo "Starting Celery Worker for KB Service..."
echo "========================================"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "ERROR: Virtual environment not found!"
    echo "Please run: python -m venv .venv"
    echo "Then: source .venv/bin/activate"
    echo "Then: pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Load environment variables if .env file exists
if [ -f ".env" ]; then
    echo "Loading environment from .env file..."
    export $(cat .env | grep -v ^# | xargs)
fi

# Start the worker with default settings
echo "Starting worker with 2 concurrent processes..."
python run_celery_worker.py --concurrency 2 --loglevel info