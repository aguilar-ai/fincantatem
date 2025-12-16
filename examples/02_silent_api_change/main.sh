#!/bin/sh

set -e

uv run fastapi dev server.py &
SERVER_PID=$!

# Run the client and always kill the server afterward
(
    uv run client.py
)
EXIT_CODE=$?

kill $SERVER_PID 2>/dev/null || true

exit $EXIT_CODE