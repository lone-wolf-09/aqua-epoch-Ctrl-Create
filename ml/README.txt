Aqua-Epoch baseline scripts

Copy these three .py files into your project's ml/ folder.
Expected CSV: ml/data/official/aqua_epoch_dataset.csv

From project root, in PowerShell:
  .\backend\venv\Scripts\Activate.ps1
  python -m pip install pandas scikit-learn joblib
  python .\ml\analyze_label_semantics.py
  python .\ml\train_classifier_baseline.py
  python .\ml\train_regression_baseline.py

Outputs:
  ml/models/official_classifier_baseline.pkl
  ml/models/official_ph_regressor_baseline.pkl
  ml/reports/classification_baseline.json
  ml/reports/regression_baseline.json

Class 0/1 meaning is unknown. Distribution comparisons against illustrative ranges are not authoritative label decoding. Do not display SAFE/UNSAFE without official documentation.

Classification uses valid binary labels and all 18 water measurements. It changes impossible pH and negative Hardness/Fluoride values to missing, then imputes from training data inside a pipeline. High TDS/turbidity values are retained pending units/data dictionary and later sensitivity analysis.

Regression uses valid pH targets in [0,14], and the other 17 measurements as inputs. It excludes both pH and Potability from inputs to prevent leakage. Both tasks use train/validation/test splits (70/15/15), seed 42, a simple dummy baseline, and Random Forest.
