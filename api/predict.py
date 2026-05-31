import json

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


def _get_request_attr(request, name, default=None):
    if hasattr(request, name):
        return getattr(request, name)
    if isinstance(request, dict):
        return request.get(name, default)
    return default


def handler(request):
    """
    Vercel Serverless Function entrypoint.

    Expects `request` to be a Vercel-provided request object (or a dict-like).
    Returns a Vercel-compatible response dict: {statusCode, headers, body}.
    """

    headers = _get_request_attr(request, "headers", {}) or {}
    method = (_get_request_attr(request, "method", "POST") or "POST").upper()

    cors_headers = {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }

    if method == "OPTIONS":
        return {"statusCode": 200, "headers": cors_headers, "body": json.dumps({"message": "OK"})}

    if method != "POST":
        return {
            "statusCode": 405,
            "headers": cors_headers,
            "body": json.dumps({"error": "Method not allowed. Use POST."}),
        }

    try:
        body = _get_request_attr(request, "body", None)
        if body is None and isinstance(request, dict):
            body = request.get("data")
        if body is None:
            return {
                "statusCode": 400,
                "headers": cors_headers,
                "body": json.dumps({"error": "Request body is required."}),
            }

        if isinstance(body, (bytes, bytearray)):
            body_text = body.decode("utf-8")
        else:
            body_text = str(body)

        input_data = json.loads(body_text)
        status_code, response_data = build_prediction_response(input_data)
        return {"statusCode": status_code, "headers": cors_headers, "body": json.dumps(response_data)}

    except json.JSONDecodeError:
        return {
            "statusCode": 400,
            "headers": cors_headers,
            "body": json.dumps({"error": "Request body must be valid JSON."}),
        }
    except ValueError as error:
        return {
            "statusCode": 400,
            "headers": cors_headers,
            "body": json.dumps({"error": str(error)}),
        }
    except Exception as error:
        request_id = headers.get("x-vercel-id") or headers.get("x-request-id")
        payload = {"error": str(error)}
        if request_id:
            payload["request_id"] = request_id
        return {"statusCode": 500, "headers": cors_headers, "body": json.dumps(payload)}
