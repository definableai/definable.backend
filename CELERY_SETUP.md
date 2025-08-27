# Celery Worker Setup Guide

This guide explains how to run Celery workers for background task processing in the KB service.

## Prerequisites

1. **Redis or RabbitMQ** - Celery requires a message broker
2. **Environment variables** - Set in `.env` file:
   ```
   CELERY_BROKER_URL=redis://localhost:6379/0
   CELERY_RESULT_BACKEND=redis://localhost:6379/0
   ```

## Quick Start

### Windows
```cmd
# Start worker with default settings
start_worker.bat

# Or run directly
python run_celery_worker.py
```

### Linux/Mac
```bash
# Make script executable
chmod +x start_worker.sh

# Start worker
./start_worker.sh

# Or run directly
python run_celery_worker.py
```

## Advanced Usage

### Basic Worker Commands

```bash
# Start worker with custom concurrency
python run_celery_worker.py --concurrency 4

# Start worker with debug logging
python run_celery_worker.py --loglevel debug

# Start worker for specific queues
python run_celery_worker.py --queues kb_processing,kb_indexing

# Start worker with autoscaling (max 8, min 2 processes)
python run_celery_worker.py --autoscale 8,2
```

### Using the Management Script

```bash
# Start worker
python scripts/celery_manager.py worker --concurrency 4

# Check status
python scripts/celery_manager.py status

# Monitor tasks in real-time
python scripts/celery_manager.py monitor

# Start Flower web monitoring
python scripts/celery_manager.py flower --port 5555

# Purge all pending tasks
python scripts/celery_manager.py purge
```

## Configuration Options

### Worker Options

| Option | Default | Description |
|--------|---------|-------------|
| `--concurrency` | 2 | Number of concurrent worker processes |
| `--loglevel` | info | Logging level (debug, info, warning, error, critical) |
| `--queues` | celery | Comma-separated list of queues to consume |
| `--hostname` | kb-worker@%h | Custom hostname for the worker |
| `--pool` | prefork | Pool implementation (prefork, eventlet, gevent, threads) |
| `--max-tasks-per-child` | 1000 | Max tasks before worker process restart |
| `--prefetch-multiplier` | 1 | Task prefetch multiplier |
| `--autoscale` | None | Enable autoscaling (format: "max,min") |

### Environment Variables

```bash
# Required
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# Database connection
DATABASE_URL=postgresql+asyncpg://user:password@localhost/definable

# Other required settings
OPENAI_API_KEY=your_openai_key
S3_BUCKET=your_s3_bucket
S3_ACCESS_KEY=your_s3_key
S3_SECRET_KEY=your_s3_secret
S3_ENDPOINT=your_s3_endpoint
```

## Task Types

The KB service processes these types of tasks:

### Document Processing Tasks
- **File Upload Processing** - Extract content from uploaded files
- **URL Content Extraction** - Scrape and extract content from URLs
- **Content Indexing** - Generate embeddings and store in vector database

### Task Queues
- `celery` (default) - General purpose queue
- `kb_processing` - Document content extraction
- `kb_indexing` - Vector embedding generation

## Production Deployment

### Systemd Service (Linux)

Create `/etc/systemd/system/celery-kb-worker.service`:

```ini
[Unit]
Description=Celery Worker for KB Service
After=network.target

[Service]
Type=notify
User=your_app_user
Group=your_app_group
WorkingDirectory=/path/to/definable.backend
Environment=PATH=/path/to/definable.backend/.venv/bin
EnvironmentFile=/path/to/definable.backend/.env
ExecStart=/path/to/definable.backend/.venv/bin/python run_celery_worker.py --concurrency 4
ExecReload=/bin/kill -s HUP $MAINPID
TimeoutStopSec=600
KillSignal=SIGTERM
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable celery-kb-worker
sudo systemctl start celery-kb-worker
sudo systemctl status celery-kb-worker
```

### Docker Deployment

```dockerfile
# Add to your Dockerfile
CMD ["python", "run_celery_worker.py", "--concurrency", "4", "--loglevel", "info"]
```

Or use docker-compose:

```yaml
services:
  celery-worker:
    build: .
    command: python run_celery_worker.py --concurrency 4
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CELERY_RESULT_BACKEND=redis://redis:6379/0
    depends_on:
      - redis
      - postgres
    volumes:
      - .:/app
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    restart: unless-stopped
```

### Multiple Workers

For high availability, run multiple workers:

```bash
# Worker 1 - General processing
python run_celery_worker.py --hostname worker1@%h --concurrency 2

# Worker 2 - High priority tasks
python run_celery_worker.py --hostname worker2@%h --queues kb_processing --concurrency 4

# Worker 3 - Indexing tasks
python run_celery_worker.py --hostname worker3@%h --queues kb_indexing --concurrency 2
```

## Monitoring

### Flower Web UI

Start Flower for web-based monitoring:

```bash
# Start Flower
python scripts/celery_manager.py flower

# Access at http://localhost:5555
```

### Command Line Monitoring

```bash
# Check worker status
python scripts/celery_manager.py status

# Monitor events
python scripts/celery_manager.py monitor

# Inspect workers
python scripts/celery_manager.py inspect
```

### Logging

Worker logs are output to stdout. For production, redirect to files:

```bash
# Log to file
python run_celery_worker.py --loglevel info > worker.log 2>&1

# Use logrotate for log management
```

## Troubleshooting

### Common Issues

1. **Worker not starting**
   - Check Redis/RabbitMQ is running
   - Verify environment variables
   - Check database connectivity

2. **Tasks failing**
   - Check worker logs
   - Verify file permissions
   - Check S3 credentials

3. **High memory usage**
   - Reduce `--max-tasks-per-child`
   - Use `--pool threads` for I/O-bound tasks
   - Monitor with `--loglevel debug`

### Debug Commands

```bash
# Test Celery connection
python -c "from common.q import celery_app; print(celery_app.control.inspect().ping())"

# Check registered tasks
python -c "from common.q import celery_app; print(celery_app.control.inspect().registered())"

# Test task execution
python -c "from tasks.kb_tasks import process_document_task; print('Tasks loaded successfully')"
```

## Performance Tuning

### Worker Configuration

- **CPU-bound tasks**: Use `--pool prefork` with `--concurrency` = CPU cores
- **I/O-bound tasks**: Use `--pool threads` with higher concurrency
- **Mixed workload**: Use separate workers for different task types

### Resource Limits

```bash
# Limit memory per worker
python run_celery_worker.py --max-memory-per-child 200000  # 200MB

# Restart workers periodically
python run_celery_worker.py --max-tasks-per-child 100

# Control task prefetching
python run_celery_worker.py --prefetch-multiplier 1
```

### Autoscaling

```bash
# Scale between 2-8 workers based on load
python run_celery_worker.py --autoscale 8,2
```

For more advanced configuration, see the [Celery documentation](https://docs.celeryproject.org/).