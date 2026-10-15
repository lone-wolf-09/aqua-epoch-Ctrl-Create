# AquaSense — ML-Based Water Quality Analysis

**Event:** Aqua-Epoch, TechFEST'26  
**Team:** Ctrl+Create  
**Institute:** Sant Longowal Institute of Engineering and Technology (SLIET), Longowal

## Team Members
- Aman
- Md. Aman
- Jayant Kumar


## Project Overview
AquaSense is a prototype web application for water-quality class prediction and pH estimation. The frontend collects water-quality measurements, sends them to a FastAPI backend, and displays model outputs.

### ML Tasks
- **Classification:** Random Forest model predicts the dataset's `Potability` class using 18 input features.
- **Regression:** Gradient Boosting Regressor estimates pH using the other 17 physicochemical features, excluding pH and `Potability`.

> **Important:** The event brief specifies Safe = 1 and Unsafe = 0. Confirm this mapping against the dataset's original documentation before interpreting model labels as real-world safety. Predictions are experimental outputs and do not replace certified laboratory testing.

## Tech Stack
- Frontend: React, Vite, Tailwind CSS
- Backend: Python, FastAPI, Pydantic, Uvicorn
- ML: scikit-learn and saved model artifacts (`.pkl`)

## Repository Structure
The exact structure can vary; the current project layout discussed by the team is:

```text
AquaSense/
├── backend/
│   ├── app/
│   ├── models/
│   │   ├── official_classifier_baseline.pkl
│   │   └── official_ph_regressor_candidate.pkl
│   └── requirements.txt
├── frontend/
│   ├── src/
│   ├── .env.example
│   ├── package.json
│   └── ...
└── ml/
    ├── data/
    ├── models/
    └── reports/
```

Do not commit virtual environments, secrets, private contact details, or local `.env` files containing credentials.

## Prerequisites
- Python 3.10+ recommended (use the version used during model training if different)
- Node.js and npm
- Git

## Run Locally

### 1. Backend (Windows PowerShell)
Open a terminal in the repository root:

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If the backend virtual environment already exists, activate it instead of recreating it. Then start the API:

```powershell
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

API docs: <http://127.0.0.1:8000/docs>

### 2. Frontend
Open a second terminal from the repository root:

```powershell
cd frontend
npm install
```

Create `frontend/.env` with:

```env
VITE_API_URL=http://127.0.0.1:8000
```

Then run:

```powershell
npm run dev
```

Open the local URL printed by Vite, usually <http://localhost:5173>.

Keep both terminals open while using the app.

## API
The frontend sends a JSON request to:

```text
POST http://127.0.0.1:8000/predict/
```

The request schema expects the 18 numeric water-quality fields used by the trained classifier. The response includes the predicted class, status label, confidence score, and estimated pH.

For exact field names and request shape, use the live API documentation at <http://127.0.0.1:8000/docs>.

## Model Evaluation Summary
Reported held-out test results from the project's current training runs:

### Classification — Random Forest
- Accuracy: **99.26%**
- Balanced accuracy: **99.24%**
- Class 1 precision: **98.94%**
- Class 1 recall: **99.16%**
- Class 1 F1-score: **99.05%**
- ROC-AUC: **0.99987**

### pH Regression — Gradient Boosting Regressor
- MAE: **0.4868 pH units**
- RMSE: **0.6519 pH units**
- R²: **0.0230**

The classification metrics are specific to the supplied held-out dataset and require independent validation. The pH regression model has weak explanatory power. Neither output is a certified water-safety assessment.

