"""
train_ats_model.py

Offline training script for the XGBoost ATS-parseability model used by
ml_models.predict_ats_score(). Run this once (or whenever you want to
retrain) from inside the project folder:

    python3 train_ats_model.py
It writes ats_model.joblib next to this script, which ml_models.py loads
lazily at request time. The server does NOT train anything at runtime —
this file is only ever run manually / offline.

See the comment above ATS_FEATURE_NAMES / _synthetic_ats_training_data in
ml_models.py for why this is trained on synthetic, heuristic-derived labels
rather than a real labeled ATS-outcome dataset (none exists publicly), and
what to do if you obtain real labeled data later.
"""

from ml_models import train_ats_xgboost_model, ats_feature_importances, ATS_MODEL_PATH

if __name__ == "__main__":
    print("Training XGBoost ATS-parseability model on synthetic data...")
    train_ats_xgboost_model()
    print(f"Saved model to {ATS_MODEL_PATH}")
    importances = ats_feature_importances()
    if importances:
        print("\nFeature importances:")
        for name, score in sorted(importances.items(), key=lambda x: -x[1]):
            print(f"  {name:28s} {score:.4f}")