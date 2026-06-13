"""TCP proxy that rewrites Host header to localhost:5000."""
import socket, threading, sys, re

TARGET_HOST = "127.0.0.1"
TARGET_PORT = 5000
LISTEN_PORT = 5001

def proxy(client):
    target = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    target.connect((TARGET_HOST, TARGET_PORT))

    # Read initial request from client
    data = client.recv(65536)
    # Rewrite Host header
    data = re.sub(
        rb"(Host|host):\s[^\r\n]+",
        b"Host: localhost:5000",
        data
    )
    target.sendall(data)

    # Bidirectional copy
    def forward(src, dst):
        try:
            while True:
                chunk = src.recv(65536)
                if not chunk:
                    break
                dst.sendall(chunk)
        except:
            pass
        finally:
            try: dst.close()
            except: pass

    t1 = threading.Thread(target=forward, args=(client, target), daemon=True)
    t2 = threading.Thread(target=forward, args=(target, client), daemon=True)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", LISTEN_PORT))
    server.listen(100)
    print(f"Proxy listening on 0.0.0.0:{LISTEN_PORT} -> {TARGET_HOST}:{TARGET_PORT}")
    while True:
        client, addr = server.accept()
        threading.Thread(target=proxy, args=(client,), daemon=True).start()

if __name__ == "__main__":
    main()
