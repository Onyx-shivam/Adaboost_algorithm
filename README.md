# Adaboost_algorithm

# AdaBoost Prediction Studio

A clean, colourful web app that serves a trained **AdaBoost** classifier (`Adaboost_algorithm.pkl`) through a **Flask** backend. Enter the input details in the form and get the predicted outcome along with a confidence score and a probability breakdown.

---

## Features

- Single-file app: backend, HTML, CSS and JavaScript all live in `app.py`
- Form fields are generated automatically from the model's column names
- Animated confidence ring and per-class probability bars
- Form progress bar, "Clear" and "Sample values" buttons
- Responsive layout (works on desktop and mobile)
- `/health` endpoint for deployment checks
- Friendly error messages for missing or invalid input

---

## Project Structure

```
Adaboost_algorithm/
|-- app.py                    # Flask app + full UI
|-- Adaboost_algorithm.pkl    # trained AdaBoost model
|-- requirements.txt          # Python dependencies
|-- README.md                 # this file
```

---

## Tech Stack

| Layer     | Tools                              |
|-----------|------------------------------------|
| Backend   | Python, Flask                      |
| ML        | scikit-learn (AdaBoostClassifier)  |
| Data      | NumPy, Pandas                      |
| Frontend  | HTML, CSS, JavaScript (in app.py)  |
| Server    | Gunicorn (for deployment)          |

---

## Run Locally (Windows)

1. Open the project folder in VS Code / terminal:

   ```
   cd C:\Users\hp\Desktop\Machine_learning\Adaboost_algorithm
   ```

2. Activate your virtual environment (if not already active):

   ```
   ..\venv\Scripts\activate
   ```

3. Install the dependencies:

   ```
   pip install -r requirements.txt
   ```

4. Make sure `Adaboost_algorithm.pkl` is in the **same folder as `app.py`**.

5. Start the app:

   ```
   python app.py
   ```

6. Open your browser at **http://127.0.0.1:5000**

---

## Customising the App

All settings are at the top of `app.py`.

### Labels, sample values and dropdowns (`META`)

The key is the column name in lowercase. Every option is optional.

```python
META = {
    "age":             {"default": 18, "step": 1, "min": 10, "max": 100},
    "high_school_gpa": {"label": "High School GPA", "default": 3.6, "step": 0.01},

    # turn a field into a dropdown (use the SAME encoding you used while training)
    "gender": {"options": [("Male", 0), ("Female", 1)], "default": 0},
}
```

| Key       | Meaning                                         |
|-----------|-------------------------------------------------|
| `label`   | Text shown above the field                      |
| `default` | Sample value pre-filled in the form             |
| `step`    | Number step (`1` for whole numbers, `0.1` etc.) |
| `min/max` | Optional limits                                 |
| `options` | List of `("Text", value)` to make a dropdown    |

### Outcome names (`CLASS_LABELS`)

Show readable names instead of `0` and `1`:

```python
CLASS_LABELS = {
    0: "Not Eligible",
    1: "Eligible",
}
```

---

## Deploy on Render

1. Push these files to a GitHub repository:
   `app.py`, `Adaboost_algorithm.pkl`, `requirements.txt`, `README.md`
2. On [Render](https://render.com), create a new **Web Service** and connect the repository.
3. Use these settings:

   | Setting        | Value                |
   |----------------|----------------------|
   | Runtime        | Python 3             |
   | Build Command  | `pip install -r requirements.txt` |
   | Start Command  | `gunicorn app:app`   |

4. Click **Deploy**. Render sets the `PORT` variable automatically and the app reads it.

---

## API

### `POST /predict`

Request body (values in the same order as the form fields):

```json
{ "values": [18, 0, 0, 60000, 3.6, 1250, 27, 95, 3] }
```

Response:

```json
{
  "label": "Eligible",
  "color": "#20C9A6",
  "confidence": 87.4,
  "probs": [
    { "label": "Not Eligible", "value": 12.6, "color": "#20C9A6" },
    { "label": "Eligible", "value": 87.4, "color": "#FF6B6B" }
  ]
}
```

### `GET /health`

Returns `{"status": "ok"}`.

---

## Troubleshooting

**`FileNotFoundError: Adaboost_algorithm.pkl`**
The model file is not next to `app.py`. Copy it into the project folder and check the exact file name (Windows can hide extensions, so make sure it is not `.pkl.pkl`).

**scikit-learn version warning or load error**
The model must be loaded with the same scikit-learn version used for training. Check it with `pip show scikit-learn` in the training environment and pin that version in `requirements.txt`, for example `scikit-learn==1.5.2`.

**"Expected N values" error**
The number of form fields does not match the model's training columns. Make sure the model was saved from a DataFrame with the same columns, in the same order.

**Wrong or confusing prediction labels**
Set `CLASS_LABELS` and use dropdown `options` that match the encoding used during training.

---

## Author

Built by **Shivam** as part of a machine learning model deployment series.