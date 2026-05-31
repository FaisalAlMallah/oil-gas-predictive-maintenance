import re
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "xgboost_pipeline.pkl"

MODEL_COLUMNS = [
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

REQUEST_REQUIRED_COLUMNS = [
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

NUMERIC_COLUMNS = [
    "vibration_rms",
    "temperature_motor",
    "current_phase_avg",
    "pressure_level",
    "rpm",
    "hours_since_maintenance",
    "ambient_temp",
]

MACHINE_ID_OPTIONS = list(range(1, 21))
MACHINE_TYPE_OPTIONS = {
    "cnc": "CNC",
    "compressor": "Compressor",
    "pump": "Pump",
    "robotic arm": "Robotic Arm",
}
MACHINE_TYPE_ID_RANGES = {
    "CNC": range(1, 6),
    "Pump": range(6, 11),
    "Compressor": range(11, 16),
    "Robotic Arm": range(16, 21),
}
MACHINE_TYPE_DEFAULT_IDS = {
    "CNC": 3,
    "Pump": 8,
    "Compressor": 13,
    "Robotic Arm": 18,
}
OPERATING_MODE_OPTIONS = {
    "idle": "idle",
    "normal": "normal",
    "peak": "peak",
}


@lru_cache(maxsize=1)
def load_model():
    return joblib.load(MODEL_PATH)


def _normalize_machine_id(value):
    if isinstance(value, str):
        match = re.search(r"\d+", value.strip())
        if not match:
            raise ValueError("machine_id must contain a valid machine number.")
        machine_id = int(match.group())
    else:
        machine_id = int(value)

    if machine_id not in MACHINE_ID_OPTIONS:
        raise ValueError("machine_id must be between 1 and 20.")

    return machine_id


def _normalize_machine_type(value):
    normalized_value = str(value).strip().lower()
    if normalized_value not in MACHINE_TYPE_OPTIONS:
        raise ValueError("machine_type must be one of: CNC, Compressor, Pump, Robotic Arm.")
    return MACHINE_TYPE_OPTIONS[normalized_value]


def _infer_machine_id(machine_type):
    return MACHINE_TYPE_DEFAULT_IDS[machine_type]


def _validate_machine_id_for_machine_type(machine_id, machine_type):
    valid_ids = MACHINE_TYPE_ID_RANGES[machine_type]
    if machine_id not in valid_ids:
        raise ValueError(
            f"machine_id must be within {valid_ids.start}-{valid_ids.stop - 1} for machine_type {machine_type}."
        )
    return machine_id


def _normalize_operating_mode(value):
    normalized_value = str(value).strip().lower()
    if normalized_value not in OPERATING_MODE_OPTIONS:
        raise ValueError("operating_mode must be one of: idle, normal, peak.")
    return OPERATING_MODE_OPTIONS[normalized_value]


def normalize_input_data(input_data):
    missing_columns = [column for column in REQUEST_REQUIRED_COLUMNS if column not in input_data]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    normalized_data = dict(input_data)
    normalized_data["machine_type"] = _normalize_machine_type(input_data["machine_type"])
    if "machine_id" in input_data and input_data["machine_id"] not in (None, ""):
        machine_id = _normalize_machine_id(input_data["machine_id"])
        normalized_data["machine_id"] = _validate_machine_id_for_machine_type(
            machine_id,
            normalized_data["machine_type"],
        )
    else:
        normalized_data["machine_id"] = _infer_machine_id(normalized_data["machine_type"])
    normalized_data["operating_mode"] = _normalize_operating_mode(input_data["operating_mode"])

    for column in NUMERIC_COLUMNS:
        normalized_data[column] = float(input_data[column])

    return normalized_data


def predict_failure(input_data):
    model = load_model()
    normalized_data = normalize_input_data(input_data)
    input_df = pd.DataFrame([normalized_data])

    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])

    return prediction, probability

if __name__ == "__main__":

    sample_machine = {
        "machine_type": "Pump",
        "vibration_rms": 0.8,
        "temperature_motor": 85,
        "current_phase_avg": 12.5,
        "pressure_level": 30,
        "rpm": 2500,
        "operating_mode": "normal",
        "hours_since_maintenance": 120,
        "ambient_temp": 12,
    }

    prediction, probability = predict_failure(sample_machine)

    print("Prediction:", prediction)
    print("Failure Probability:", probability)

    if prediction == 1:
        print("Result: Failure likely within 24 hours")
    else:
        print("Result: No failure predicted within 24 hours")
