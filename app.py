 

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

# The scaler was pickled with scikit-learn 1.6.1. Newer 1.x releases load it
# safely; suppress only that version-mismatch notice to keep API logs clean.
try:
    from sklearn.exceptions import InconsistentVersionWarning

    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:  # pragma: no cover - compatibility with very old sklearn
    pass


# ---------------------------------------------------------------------------
# Keras backend selection
# ---------------------------------------------------------------------------
# A .keras file created by Keras 3 is backend-neutral: the same architecture and
# weights can be executed by TensorFlow or by another Keras 3 backend. We prefer
# TensorFlow when it imports successfully. If the TensorFlow native libraries are
# unavailable/blocked (for example by Windows application-control policy), Keras
# automatically falls back to the Torch backend so the exact same model still runs.
def _tensorflow_imports() -> bool:
    """Return True if TensorFlow can be imported in this interpreter."""
    try:
        completed = subprocess.run(
            [sys.executable, "-c", "import tensorflow as tf; assert tf.__version__"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )
        return completed.returncode == 0
    except Exception:
        return False


def _configure_keras_backend() -> str:
    """Set KERAS_BACKEND before importing keras and return the chosen backend."""
    requested = (
        os.environ.get("PLACEMENT_KERAS_BACKEND")
        or os.environ.get("KERAS_BACKEND")
        or ""
    ).strip().lower()

    if requested:
        backend = requested
    elif _tensorflow_imports():
        backend = "tensorflow"
    else:
        backend = "torch"

    os.environ["KERAS_BACKEND"] = backend
    return backend


KERAS_BACKEND = _configure_keras_backend()

import keras  # noqa: E402  (must be imported after KERAS_BACKEND is set)


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "student_placement_model.keras"
SCALER_PATH = MODEL_DIR / "placement.pkl"

# Fallback is used only if an older scaler lacks feature_names_in_. It matches
# the inspected training dataset exactly and must not be reordered.
FALLBACK_FEATURES: Tuple[str, ...] = (
    "CGPA",
    "10th_Percentage",
    "12th_Percentage",
    "Internships",
    "Projects",
    "Aptitude_Score",
    "Communication_Skills",
    "Technical_Skills",
    "Backlogs",
)

# Presentation/validation metadata only. These values are NOT model features.
# Ranges reflect the scale of the training data and common real-world scales.
FEATURE_CONFIG: Dict[str, Dict[str, Any]] = {
    "CGPA": {
        "label": "CGPA",
        "unit": "/ 10",
        "min": 0.0,
        "max": 10.0,
        "step": 0.01,
        "placeholder": "e.g. 8.20",
        "hint": "Cumulative grade point average on a 10-point scale.",
    },
    "10th_Percentage": {
        "label": "Class 10 Percentage",
        "unit": "%",
        "min": 0.0,
        "max": 100.0,
        "step": 0.1,
        "placeholder": "e.g. 85.5",
        "hint": "Percentage scored in class 10.",
    },
    "12th_Percentage": {
        "label": "Class 12 Percentage",
        "unit": "%",
        "min": 0.0,
        "max": 100.0,
        "step": 0.1,
        "placeholder": "e.g. 78.0",
        "hint": "Percentage scored in class 12.",
    },
    "Internships": {
        "label": "Internships",
        "unit": "count",
        "min": 0,
        "max": 10,
        "step": 1,
        "placeholder": "e.g. 2",
        "hint": "Number of completed internships.",
    },
    "Projects": {
        "label": "Projects",
        "unit": "count",
        "min": 0,
        "max": 20,
        "step": 1,
        "placeholder": "e.g. 3",
        "hint": "Number of academic or personal projects.",
    },
    "Aptitude_Score": {
        "label": "Aptitude Score",
        "unit": "pts",
        "min": 0,
        "max": 100,
        "step": 1,
        "placeholder": "e.g. 72",
        "hint": "Aptitude test score out of 100.",
    },
    "Communication_Skills": {
        "label": "Communication Skills",
        "unit": "/ 10",
        "min": 1,
        "max": 10,
        "step": 1,
        "placeholder": "e.g. 8",
        "hint": "Communication skill rating from 1 to 10.",
    },
    "Technical_Skills": {
        "label": "Technical Skills",
        "unit": "/ 10",
        "min": 1,
        "max": 10,
        "step": 1,
        "placeholder": "e.g. 7",
        "hint": "Technical skill rating from 1 to 10.",
    },
    "Backlogs": {
        "label": "Backlogs",
        "unit": "count",
        "min": 0,
        "max": 10,
        "step": 1,
        "placeholder": "e.g. 0",
        "hint": "Number of active or pending backlogs.",
    },
}


# ---------------------------------------------------------------------------
# Load trained artifacts at startup (no training code is executed here)
# ---------------------------------------------------------------------------
def _load_artifacts() -> Tuple[Any, Any, Tuple[str, ...]]:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not SCALER_PATH.is_file():
        raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")

    # compile=False loads architecture + trained weights without rebuilding the
    # optimizer state. This keeps the model architecture and weights untouched.
    loaded_model = keras.models.load_model(MODEL_PATH, compile=False)

    with SCALER_PATH.open("rb") as file:
        loaded_scaler = pickle.load(file)

    stored_names = getattr(loaded_scaler, "feature_names_in_", None)
    if stored_names is not None:
        feature_names = tuple(str(name) for name in stored_names)
    else:
        feature_names = FALLBACK_FEATURES

    expected = int(loaded_model.input_shape[-1])
    if len(feature_names) != expected:
        raise ValueError(
            f"Scaler expects {len(feature_names)} features but the ANN input "
            f"layer expects {expected}. Check the model/scaler pair."
        )

    return loaded_model, loaded_scaler, feature_names


model, scaler, FEATURE_NAMES = _load_artifacts()
EXPECTED_FEATURE_COUNT = len(FEATURE_NAMES)
DENSE_UNITS = [
    int(layer.units) for layer in model.layers if hasattr(layer, "units")
]


def _build_feature_metadata() -> List[Dict[str, Any]]:
    """Create template metadata while preserving scaler feature order."""
    metadata: List[Dict[str, Any]] = []
    for index, name in enumerate(FEATURE_NAMES, start=1):
        config = FEATURE_CONFIG.get(
            name,
            {
                "label": name.replace("_", " "),
                "unit": "",
                "min": 0,
                "max": 100,
                "step": 1,
                "placeholder": "Enter value",
                "hint": "Model input feature.",
            },
        )
        metadata.append(
            {
                "index": index,
                "name": name,
                "label": config["label"],
                "unit": config["unit"],
                "min": config["min"],
                "max": config["max"],
                "step": config["step"],
                "placeholder": config["placeholder"],
                "hint": config["hint"],
            }
        )
    return metadata


FEATURES = _build_feature_metadata()

app = Flask(__name__)
app.json.sort_keys = False


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _extract_payload() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Accept JSON (frontend) and form-encoded (legacy) requests."""
    if request.is_json:
        data = request.get_json(silent=True)
        if data is None:
            return None, "Request body must contain valid JSON."
        if not isinstance(data, dict):
            return None, "JSON body must be an object."
        return data, None

    if request.form:
        return request.form.to_dict(), None

    return None, "No input data received."


def _validate_payload(
    data: Dict[str, Any],
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """Validate and convert the request into the exact training feature order."""
    missing = [
        name
        for name in FEATURE_NAMES
        if name not in data or str(data[name]).strip() == ""
    ]
    if missing:
        return None, {
            "error": "Missing required feature values.",
            "missing_fields": missing,
        }

    # Reject unknown fields so no invented feature can silently enter the model.
    extra = [key for key in data if key not in FEATURE_NAMES]
    if extra:
        return None, {
            "error": "Unknown fields are not accepted by this model.",
            "unknown_fields": extra,
        }

    values: List[float] = []
    for name in FEATURE_NAMES:
        raw_value = data[name]
        if isinstance(raw_value, bool):
            return None, {
                "error": f"'{name}' must be numeric.",
                "invalid_field": name,
            }

        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return None, {
                "error": f"'{name}' must be a number.",
                "invalid_field": name,
            }

        if not np.isfinite(value):
            return None, {
                "error": f"'{name}' must be a finite number.",
                "invalid_field": name,
            }

        config = FEATURE_CONFIG.get(name, {})
        minimum = config.get("min")
        maximum = config.get("max")
        if minimum is not None and maximum is not None and not (
            minimum <= value <= maximum
        ):
            return None, {
                "error": f"'{name}' must be between {minimum} and {maximum}.",
                "invalid_field": name,
            }

        values.append(value)

    return np.asarray([values], dtype=np.float64), None


def _summarize_features(values: np.ndarray) -> List[Dict[str, Any]]:
    row = values.reshape(-1)
    return [
        {
            "name": name,
            "label": FEATURE_CONFIG.get(name, {}).get("label", name.replace("_", " ")),
            "value": float(row[index]),
        }
        for index, name in enumerate(FEATURE_NAMES)
    ]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home() -> str:
    """Render the responsive single-page application."""
    return render_template(
        "index.html",
        features=FEATURES,
        feature_names=list(FEATURE_NAMES),
        feature_count=EXPECTED_FEATURE_COUNT,
        dense_units=DENSE_UNITS,
        keras_backend=KERAS_BACKEND,
    )


@app.route("/health")
def health() -> Any:
    """Report backend/model readiness for monitoring and debugging."""
    return jsonify(
        {
            "status": "healthy" if model is not None and scaler is not None else "degraded",
            "model_loaded": model is not None,
            "scaler_loaded": scaler is not None,
            "keras_backend": KERAS_BACKEND,
            "model_input_shape": list(model.input_shape),
            "expected_features": EXPECTED_FEATURE_COUNT,
            "features": list(FEATURE_NAMES),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


@app.route("/predict", methods=["POST"])
def predict() -> Any:
    """Scale validated input, run ANN inference, and return a JSON result."""
    data, extraction_error = _extract_payload()
    if extraction_error:
        return jsonify({"success": False, "error": extraction_error}), 400

    values, validation_error = _validate_payload(data or {})
    if validation_error:
        return jsonify({"success": False, **validation_error}), 400

    # DataFrame with named columns prevents scikit-learn's feature-name warning
    # and guarantees the scaler receives columns in the training order.
    input_frame = pd.DataFrame(values, columns=FEATURE_NAMES)

    try:
        scaled_input = scaler.transform(input_frame)
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        return jsonify(
            {"success": False, "error": "Feature scaling failed.", "detail": str(exc)}
        ), 500

    try:
        raw_output = model.predict(scaled_input, verbose=0)
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        return jsonify(
            {"success": False, "error": "Model inference failed.", "detail": str(exc)}
        ), 500

    probability = float(np.asarray(raw_output).reshape(-1)[0])
    if not np.isfinite(probability):
        return jsonify(
            {"success": False, "error": "The model returned a non-finite result."}
        ), 500

    # Sigmoid output is already a probability in [0, 1]. Clamp only for numeric
    # safety; the architecture and weights are unchanged.
    probability = float(np.clip(probability, 0.0, 1.0))
    threshold = 0.5
    prediction = "Placed" if probability >= threshold else "Not Placed"

    return jsonify(
        {
            "success": True,
            "prediction": prediction,
            "probability": round(probability, 6),
            "probability_percent": round(probability * 100, 2),
            "threshold": threshold,
            "features": _summarize_features(values),
            "disclaimer": (
                "This is a model-generated estimate based on the supplied inputs. "
                "It is not a guarantee of actual placement."
            ),
            "model": {
                "input_features": EXPECTED_FEATURE_COUNT,
                "dense_units": DENSE_UNITS,
                "preprocessing": "StandardScaler (fitted, loaded from placement.pkl)",
            },
        }
    )


@app.errorhandler(404)
def not_found(_error: Any) -> Any:
    return jsonify({"success": False, "error": "Endpoint not found."}), 404


@app.errorhandler(405)
def method_not_allowed(_error: Any) -> Any:
    return jsonify({"success": False, "error": "Method not allowed."}), 405


@app.errorhandler(500)
def internal_error(_error: Any) -> Any:
    return jsonify({"success": False, "error": "Internal server error."}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
