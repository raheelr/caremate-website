FROM python:3.11-slim

WORKDIR /app

# Install only the API dependencies (not ingestion-heavy packages)
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copy application code (only what the API needs)
COPY agents/ agents/
COPY api/ api/
COPY safety/ safety/
COPY db/__init__.py db/__init__.py
COPY db/database.py db/database.py

# Railway injects PORT
ENV PORT=8000
EXPOSE $PORT

CMD ["/bin/sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
