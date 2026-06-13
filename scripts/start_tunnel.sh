#!/bin/bash
set -e

TUNNEL_LOG=/tmp/serveo.log
PROXY_LOG=/tmp/proxy.log
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROXY_PORT=5002

pkill -f "serveo.net" 2>/dev/null || true
fuser -k "$PROXY_PORT/tcp" 2>/dev/null || true
sleep 1

# Start Host-header rewriting proxy
python3 -u -c "
import socketserver, http.server, urllib.request

TARGET = 'http://127.0.0.1:5000'

class Proxy(http.server.BaseHTTPRequestHandler):
    def do_all(self):
        body = self.rfile.read(int(self.headers.get('content-length', 0))) if self.headers.get('content-length') else None
        url = TARGET + self.path
        req = urllib.request.Request(url, data=body,
            headers={k: v for k, v in self.headers.items() if k.lower() not in ('host', 'content-length')},
            method=self.command)
        req.headers['Host'] = '127.0.0.1:5000'
        try:
            resp = urllib.request.urlopen(req)
            self.send_response(resp.status)
            for k, v in resp.headers.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(resp.read())
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.end_headers()
            self.wfile.write(e.read())
        except Exception as e:
            self.send_error(502, str(e))
    def do_GET(self): self.do_all()
    def do_POST(self): self.do_all()
    def do_PUT(self): self.do_all()
    def do_DELETE(self): self.do_all()

with socketserver.TCPServer(('0.0.0.0', $PROXY_PORT), Proxy) as httpd:
    httpd.serve_forever()
" > "$PROXY_LOG" 2>&1 &
PROXY_PID=$!
echo "Proxy PID=$PROXY_PID on port $PROXY_PORT"

# Start serveo tunnel
nohup ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 \
    -R 80:localhost:$PROXY_PORT serveo.net > "$TUNNEL_LOG" 2>&1 &
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
echo "Tunnel PID: $TUNNEL_PID"

# Inject URL into notebooks
for NB in "$PROJECT_DIR/kaggle/twitter_training/twitter_training.ipynb" "$PROJECT_DIR/kaggle/amazon_training/amazon_training.ipynb"; do
    sed -i "s|MLFLOW_TRACKING_URI=https://[a-f0-9]*-[0-9-]*\.serveousercontent\.com|MLFLOW_TRACKING_URI=$URL|g" "$NB"
    echo "Updated: $NB"
done

echo "Done. MLflow → proxy(:$PROXY_PORT) → serveo tunnel → $URL"
