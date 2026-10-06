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

        from src.conformal.predictor import predict_conformal
        from src.conformal.routing import FINAL_BENIGN, FINAL_MALICIOUS, NEEDS_STAGE2
        
        try:
            result = predict_conformal(url)
            is_safe = result['routing'] == FINAL_BENIGN
            
            # Use max(p-value) if available for confidence, or derive from proba
            if 'conformal_pvalue_benign' in result:
                p_benign = result['conformal_pvalue_benign']
                p_malicious = result['conformal_pvalue_malicious']
                conf_val = max(p_benign, p_malicious) * 100
            else:
                conf_val = result.get('confidence', 0.0) * 100
                
            response_data = {
                'url': url,
                'result': result['routing'],  # 'FINAL_BENIGN', 'FINAL_MALICIOUS', 'NEEDS_STAGE2'
                'probability': result.get('malicious_probability', 0.0),
                'confidence': conf_val,
                'details': result
            }
        except Exception as e:
            # Fallback on error
            import traceback
            traceback.print_exc()
            response_data = {
                'url': url,
                'result': 'ERROR',
                'message': str(e)
            }

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(response_data).encode('utf-8'))

if __name__ == '__main__':
    server = HTTPServer(('127.0.0.1', PORT), SecurityHandler)
    print(f"Sahi Hai Python API running on http://127.0.0.1:{PORT}")
    server.serve_forever()
