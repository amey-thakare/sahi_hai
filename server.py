"""
Optional Sahi Hai Local REST Backend (Python 3)
For advanced full-page WHOIS and online indexing if desired.
"""
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import urllib.parse
import re

PORT = 8000

class SecurityHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length).decode('utf-8')
        params = urllib.parse.parse_qs(body)
        url = params.get('url', [''])[0]

        # Simple check
        is_safe = True
        if re.search(r'(\.tk|\.ml|\.ga|\.cf|\.gq|login.*\.php|verify.*account)', url, re.I):
            is_safe = False

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps({'url': url, 'result': 'SAFE' if is_safe else 'PHISHING'}).encode('utf-8'))

if __name__ == '__main__':
    server = HTTPServer(('127.0.0.1', PORT), SecurityHandler)
    print(f"Sahi Hai Python API running on http://127.0.0.1:{PORT}")
    server.serve_forever()
