#!/bin/bash
set -e

TUNNEL_LOG=/tmp/serveo.log
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

pkill -f "serveo.net" 2>/dev/null || true

nohup ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 \
    -R 80:localhost:5000 serveo.net > "$TUNNEL_LOG" 2>&1 &
TUNNEL_PID=$!

echo "Waiting for tunnel URL..."
URL=""
for i in $(seq 1 30); do
    URL=$(grep -oP 'https://[a-f0-9]+-[0-9-]+\.serveousercontent\.com' "$TUNNEL_LOG" 2>/dev/null | head -1)
    if [ -n "$URL" ]; then
        break
    fi
    sleep 1
done

if [ -z "$URL" ]; then
    echo "Timed out waiting for tunnel"
    exit 1
fi

echo "Tunnel URL: $URL"
echo "PID: $TUNNEL_PID"

# Inject URL into notebooks
for NB in "$PROJECT_DIR/kaggle/twitter_training/twitter_training.ipynb" "$PROJECT_DIR/kaggle/amazon_training/amazon_training.ipynb"; do
    sed -i "s|MLFLOW_TRACKING_URI=https://[a-f0-9]*-[0-9-]*\.serveousercontent\.com|MLFLOW_TRACKING_URI=$URL|g" "$NB"
    echo "Updated: $NB"
done

echo "Done. Tunnel PID=$TUNNEL_PID"
