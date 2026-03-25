FROM python:3.11-slim

WORKDIR /app
COPY app/ /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')" || exit 1

EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]