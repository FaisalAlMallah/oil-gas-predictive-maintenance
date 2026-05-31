import json
from http.server import BaseHTTPRequestHandler

from src.inference import REQUEST_REQUIRED_COLUMNS, predict_failure


def build_prediction_response(input_data):
    missing_columns = [
        column for column in REQUEST_REQUIRED_COLUMNS if column not in input_data
    ]

    if missing_columns:
        return (
            400,
            {
                "error": "Missing required columns",
                "missing_columns": missing_columns,
            },
        )

    prediction, probability = predict_failure(input_data)

    if prediction == 1:
        result = "Failure likely within 24 hours"
    else:
        result = "No failure predicted within 24 hours"

    return (
        200,
        {
            "prediction": prediction,
            "failure_probability": probability,
            "failure_probability_percent": round(probability * 100, 2),
            "result": result,
        },
    )


class handler(BaseHTTPRequestHandler):
    def _send_response(self, status_code, data):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def do_OPTIONS(self):
        self._send_response(200, {"message": "OK"})

    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            input_data = json.loads(body)
            status_code, response_data = build_prediction_response(input_data)
            return self._send_response(status_code, response_data)

        except ValueError as error:
            return self._send_response(400, {"error": str(error)})
        except Exception as e:
            return self._send_response(
                500,
                {
                    "error": str(e),
                },
            )
