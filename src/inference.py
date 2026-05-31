import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "xgboost_pipeline.pkl"

def load_model():
    model = joblib.load(MODEL_PATH)
    return model

def predict_failure(input_data):
    model = load_model()

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return prediction, probability

if __name__ == "__main__":

    sample_machine = {
        "machine_id": "M001",
        "machine_type": "Pump",
        "vibration_rms": 0.8,
        "temperature_motor": 85,
        "current_phase_avg": 12.5,
        "pressure_level": 30,
        "rpm": 2500,
        "operating_mode": "peak",
        "hours_since_maintenance": 120,
        "ambient_temp": 35
    }

    prediction, probability = predict_failure(sample_machine)

    print("Prediction:", prediction)
    print("Failure Probability:", probability)

    if prediction == 1:
        print("Result: Failure likely within 24 hours")
    else:
        print("Result: No failure predicted within 24 hours")