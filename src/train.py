import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from preprocess import load_and_preprocess
import numpy as np, joblib

def train_models():
    X_tr, X_te, y_tr, y_te, cols = load_and_preprocess()

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(
                                   n_estimators=100,
                                   max_depth=6,
                                   random_state=42
                               )
    }

    results = {}
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_tr, y_tr, cv=5, scoring="accuracy")
        print(f"\n{name}")
        print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        model.fit(X_tr, y_tr)
        results[name] = model

    joblib.dump(results["Random Forest"], "models/rf_model.pkl")
    joblib.dump(results["Logistic Regression"], "models/lr_model.pkl")
    print("\nModels saved.")
    return results, X_tr, X_te, y_tr, y_te, cols

if __name__ == "__main__":
    train_models()