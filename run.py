"""Start the server: python run.py

Separate entry point because on Windows asyncio defaults to ProactorEventLoop, which async psycopg
cannot use; the selector loop policy must be set before uvicorn starts. No-op on Linux/Docker.
"""

import asyncio
import os
import sys

import uvicorn

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

if __name__ == "__main__":
    # loop="none": keep uvicorn from picking Proactor on Windows so the policy above applies.
    # docker-compose sets HOST=0.0.0.0 so the container listens on all interfaces.
    uvicorn.run("app.main:app", host=os.environ.get("HOST", "127.0.0.1"), port=8000, loop="none")
