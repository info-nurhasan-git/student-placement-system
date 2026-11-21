# PlacePredict AI — Student Placement Prediction Web Application

A complete Flask web application that serves an **existing trained ANN model** and an **existing fitted StandardScaler**. The model is never retrained, and its architecture and weights are loaded exactly as saved.

---

## 1. What was inspected

No original training script was present in the workspace, so the input contract was recovered directly from the trained artifacts and the available dataset:

- `student_placement_model.keras`
  - Keras 3 `Sequential` model.
  - Input shape: `(None, 9)`.
  - Layers: `Dense(16, relu) → Dense(8, relu) → Dense(1, sigmoid)`.
- `placement.pkl`
  - `sklearn.preprocessing.StandardScaler`.
  - `n_features_in_ = 9`.
  - `feature_names_in_` (exact order):

| # | Feature | Meaning |
|---|---------|---------|
| 1 | `CGPA` | Cumulative grade point average |
| 2 | `10th_Percentage` | Class 10 percentage |
| 3 | `12th_Percentage` | Class 12 percentage |
| 4 | `Internships` | Number of internships |
| 5 | `Projects` | Number of projects |
| 6 | `Aptitude_Score` | Aptitude test score |
| 7 | `Communication_Skills` | Communication rating |
| 8 | `Technical_Skills` | Technical rating |
| 9 | `Backlogs` | Number of backlogs |

The dataset also contains `Student_ID` (identifier) and `Placement` (target). Neither is an input feature.

Because the final layer is `sigmoid`, the raw model output is a probability. The app applies a standard `0.5` threshold:

- `probability >= 0.5` → **Placed**
- `probability < 0.5` → **Not Placed**

> The displayed probability is a model estimate, not a guarantee of actual placement.

---

## 2. Folder structure and file purposes

```text
Student-Placement-Prediction/
│
├── app.py                          # Flask app, REST API, validation, model/scaler loading
├── requirements.txt                # Python dependencies
├── README.md                       # This documentation
│
├── model/
│   ├── student_placement_model.keras  # Existing trained ANN (copied unchanged)
│   └── placement.pkl                  # Existing fitted StandardScaler (copied unchanged)
│
├── templates/
│   └── index.html                  # Homepage, input form and result dashboard
│
└── static/
    ├── css/
    │   └── style.css               # Professional responsive styling
    ├── js/
    │   └── script.js               # Validation, API calls, Chart.js, history, reports
    └── images/
        ├── favicon.svg             # Browser tab icon
        └── hero-illustration.svg   # Homepage neural-network illustration
```

### File-by-file explanation

| File | Purpose |
|------|---------|
| `app.py` | Loads the `.keras` model and `.pkl` scaler once at startup. Defines `/`, `/predict` and `/health`. Reads feature names from the scaler, validates incoming values, builds a one-row pandas DataFrame in the exact training order, applies `scaler.transform()`, runs `model.predict()`, and returns JSON. |
| `requirements.txt` | Lists Flask, NumPy, pandas, scikit-learn, Keras, TensorFlow and Torch (Keras backend fallback). |
| `model/student_placement_model.keras` | Your trained ANN. Copied byte-for-byte; not retrained or modified. |
| `model/placement.pkl` | Your fitted scaler. Copied byte-for-byte; used with `transform()` only. |
| `templates/index.html` | Single-page UI: navbar, hero, project/model/workflow sections, dynamically generated input form (from backend metadata), result dashboard, history table and footer. |
| `static/css/style.css` | Custom design system: gradients, cards, responsive grid, loading overlay, chart layout, print styles and mobile breakpoints. |
| `static/js/script.js` | Client-side validation, fetch requests, error handling, Chart.js doughnut chart, input summary rendering, localStorage history, report download and health indicator. |
| `static/images/favicon.svg` | Small neural-network brand mark used in the navbar and browser tab. |
| `static/images/hero-illustration.svg` | Vector illustration of the student profile flowing through the ANN to a placement result. |

---

## 3. Setup and run

### 3.1 Create a virtual environment

```powershell
cd "C:\Users\nurha\OneDrive\Desktop\Code\DeepLearningProject\Student-Placement-Prediction"
python -m venv --system-site-packages .venv
```

`--system-site-packages` lets the project reuse the already-installed Keras/Torch/scikit-learn stack while Flask is installed inside the project.

### 3.2 Install dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3.3 Start the server

**Easiest option (Windows):** double-click `START_APP.bat`. It starts the server and opens the browser automatically.

Manual option:

```powershell
.\.venv\Scripts\python.exe app.py
```

Open <http://127.0.0.1:5000>.

> **Important:** Do **not** open `templates/index.html` directly in Chrome/Edge and do **not** use VS Code “Live Server” for this project. The template contains Jinja expressions such as `{{ url_for(...) }}`, which are replaced by Flask only when the app is served from `http://127.0.0.1:5000`. Opening the file directly will make CSS and JavaScript appear broken.

Optional environment variables:

```powershell
$env:FLASK_DEBUG = "1"                 # enable debug reloader
$env:PLACEMENT_KERAS_BACKEND = "torch"  # or "tensorflow"
```

---

## 4. Backend selection note

Keras 3 `.keras` files are backend-neutral. This app prefers TensorFlow when TensorFlow imports successfully (it does in the project virtual environment, so the default backend is `tensorflow`). If TensorFlow native libraries are unavailable or blocked on another machine, the app automatically falls back to the **Torch backend for Keras**. The model architecture, layer names, weights, input order and sigmoid output remain exactly the same — only the execution backend changes.

You can force a backend with `PLACEMENT_KERAS_BACKEND`, for example:

```powershell
$env:PLACEMENT_KERAS_BACKEND = "torch"
```

---

## 5. REST API

### `GET /`

Returns the homepage with form fields generated from the scaler's feature metadata.

### `GET /health`

Example response:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "scaler_loaded": true,
  "keras_backend": "torch",
  "model_input_shape": [null, 9],
  "expected_features": 9,
  "features": ["CGPA", "10th_Percentage", "..."],
  "timestamp": "2026-11-21T..."
}
```

### `POST /predict`

Accepts JSON (or form-encoded data):

```json
{
  "CGPA": 8.2,
  "10th_Percentage": 51.3,
  "12th_Percentage": 63.8,
  "Internships": 1,
  "Projects": 1,
  "Aptitude_Score": 48,
  "Communication_Skills": 9,
  "Technical_Skills": 2,
  "Backlogs": 3
}
```

Success response (abbreviated):

```json
{
  "success": true,
  "prediction": "Not Placed",
  "probability": 0.000031,
  "probability_percent": 0.0,
  "threshold": 0.5,
  "features": [{"name": "CGPA", "label": "CGPA", "value": 8.2}],
  "disclaimer": "This is a model-generated estimate...",
  "model": {"input_features": 9, "dense_units": [16, 8, 1]}
}
```

Validation errors return HTTP `400` with `success: false` and a descriptive `error` field.

---

## 6. Additional features implemented

- Reset form and “Use Sample” button.
- Per-field and API-level validation.
- Loading overlay and button spinner.
- Inline error/success alerts.
- Chart.js probability doughnut chart.
- Student input summary table.
- Browser-local prediction history (last 5) with reload and clear actions.
- Download prediction report as JSON.
- Print-friendly result dashboard.
- Fully responsive layout for mobile, tablet and desktop.

---

## 7. Testing checklist

1. `GET /health` → `status: healthy`.
2. Submit the sample values → dashboard renders prediction, probability and chart.
3. Submit an empty field → highlighted validation error.
4. Submit an out-of-range value → validation error.
5. Reset → form and dashboard return to the initial state.
6. Reload the page → recent prediction history is restored.
7. Download report → JSON file opens correctly.

---

## 8. Troubleshooting: CSS or JavaScript not working

| Symptom | Cause | Fix |
|---------|-------|-----|
| Page looks unstyled | The HTML file was opened directly from `templates/` | Run `START_APP.bat` or `python app.py`, then visit `http://127.0.0.1:5000` |
| Browser shows `{{ url_for(...) }}` text | Flask did not render the template | Same as above; never use Live Server for Jinja templates |
| Chart missing | Internet blocked CDN for Chart.js | The numeric probability still displays; allow `cdn.jsdelivr.net` or host Chart.js locally |
| Buttons do nothing | JS did not load because of the same Jinja issue | Use the Flask URL, then hard-refresh with `Ctrl + F5` |
| Health badge says offline | Flask server is not running | Start the server and refresh |
