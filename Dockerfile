# One image for both services (API and MCP server); they differ only by the command in docker-compose.yml.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

# Dependencies first: this layer is cached and rebuilt only when requirements.txt changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app app
COPY mcp_server mcp_server
COPY scripts scripts
COPY eval eval
COPY run.py .

EXPOSE 8000 8001

# Runs the API by default; the MCP service overrides the command in docker-compose.yml.
CMD ["python", "run.py"]
