import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from api.predict import build_prediction_response

ROOT_DIR = Path(__file__).resolve().parent


class AppHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT_DIR), **kwargs)

    def _send_json(self, status_code, data):
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self):
        parsed_path = urlparse(self.path)
        if parsed_path.path == "/api/predict":
            self._send_json(200, {"message": "OK"})
            return
        self.send_error(404, "Not Found")

    def do_GET(self):
        parsed_path = urlparse(self.path)
        if parsed_path.path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        parsed_path = urlparse(self.path)
        if parsed_path.path != "/api/predict":
            self.send_error(404, "Not Found")
            return

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            request_body = self.rfile.read(content_length)
            input_data = json.loads(request_body)
            status_code, response_data = build_prediction_response(input_data)
            self._send_json(status_code, response_data)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Request body must be valid JSON."})
        except Exception as error:
            self._send_json(500, {"error": str(error)})


def run_server(port=8000):
    server = ThreadingHTTPServer(("127.0.0.1", port), AppHandler)
    print(f"Serving predictive maintenance app at http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
