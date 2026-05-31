import json
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import joblib
import pandas as pd



MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "xgboost_pipeline.pkl"


model = joblib.load(MODEL_PATH)


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

            
            required_columns = [
                "machine_id",
                "machine_type",
                "vibration_rms",
                "temperature_motor",
                "current_phase_avg",
                "pressure_level",
                "rpm",
                "operating_mode",
                "hours_since_maintenance",
                "ambient_temp",
            ]

            # Check for missing columns
            missing_columns = [
                col for col in required_columns if col not in input_data
            ]

            if missing_columns:
                return self._send_response(
                    400,
                    {
                        "error": "Missing required columns",
                        "missing_columns": missing_columns,
                    },
                )

            # Convert numeric values
            numeric_columns = [
                "vibration_rms",
                "temperature_motor",
                "current_phase_avg",
                "pressure_level",
                "rpm",
                "hours_since_maintenance",
                "ambient_temp",
            ]

            for col in numeric_columns:
                input_data[col] = float(input_data[col])

            # Create DataFrame
            input_df = pd.DataFrame([input_data])

            # Make prediction
            prediction = int(model.predict(input_df)[0])
            probability = float(model.predict_proba(input_df)[0][1])

            if prediction == 1:
                result = "Failure likely within 24 hours"
            else:
                result = "No failure predicted within 24 hours"

            # Return result
            return self._send_response(
                200,
                {
                    "prediction": prediction,
                    "failure_probability": probability,
                    "failure_probability_percent": round(probability * 100, 2),
                    "result": result,
                },
            )

        except Exception as e:
            return self._send_response(
                500,
                {
                    "error": str(e),
                },
            )