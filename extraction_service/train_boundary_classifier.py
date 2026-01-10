"""
Train Random Forest classifier for boundary detection.

Pipeline:
1. Load feature data from extract_boundary_training_data.py
2. Split into train/validation/test sets (60/20/20)
3. Train Random Forest with hyperparameter tuning
4. Evaluate on test set
5. Save trained model

If Random Forest achieves <95% accuracy, train CNN instead.
"""

import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib

def load_training_data():
    """Load features from extract_boundary_training_data.py output."""
    data_path = Path('/home/cody/git/src/github.com/codyseavey/sudoku/extraction_service/training_data/boundary_features.npz')
    data = np.load(data_path, allow_pickle=True)

    X = data['X']
    y = data['y']
    feature_names = data['feature_names']

    print(f"Loaded dataset:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Positive rate: {np.mean(y):.2%}")

    return X, y, feature_names

def train_random_forest(X_train, y_train, X_val, y_val, feature_names):
    """Train Random Forest with hyperparameter tuning."""
    print("\nTraining Random Forest...")

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', None]
    }

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=2)
    grid_search.fit(X_train_scaled, y_train)

    best_rf = grid_search.best_estimator_
    print(f"\nBest parameters: {grid_search.best_params_}")

    # Evaluate
    y_pred = best_rf.predict(X_val_scaled)
    y_proba = best_rf.predict_proba(X_val_scaled)[:, 1]

    print("\nValidation Performance:")
    print(classification_report(y_val, y_pred, target_names=['Non-boundary', 'Boundary']))
    print(f"ROC-AUC: {roc_auc_score(y_val, y_proba):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    # Feature importance
    importances = best_rf.feature_importances_
    indices = np.argsort(importances)[::-1][:15]

    print("\nTop 15 Most Important Features:")
    for i, idx in enumerate(indices):
        print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

    return best_rf, scaler, grid_search.best_score_

def main():
    """Main training pipeline."""
    # Load data
    X, y, feature_names = load_training_data()

    # Split: 60% train, 20% validation, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

    print(f"Split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")

    # Train Random Forest
    best_rf, scaler, val_score = train_random_forest(X_train, y_train, X_val, y_val, feature_names)

    # Test set evaluation
    X_test_scaled = scaler.transform(X_test)
    y_test_pred = best_rf.predict(X_test_scaled)
    y_test_proba = best_rf.predict_proba(X_test_scaled)[:, 1]

    print("\nTest Set Performance:")
    print(classification_report(y_test, y_test_pred, target_names=['Non-boundary', 'Boundary']))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_test_proba):.4f}")

    # Save model
    output_dir = Path('/home/cody/git/src/github.com/codyseavey/sudoku/extraction_service/models')
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_rf, output_dir / 'boundary_classifier_rf.pkl')
    joblib.dump(scaler, output_dir / 'boundary_scaler.pkl')

    print(f"\nModel saved:")
    print(f"  Classifier: {output_dir / 'boundary_classifier_rf.pkl'}")
    print(f"  Scaler: {output_dir / 'boundary_scaler.pkl'}")

    # Check if performance is sufficient
    if val_score < 0.95:
        print(f"\nWARNING: Validation F1 score {val_score:.4f} < 0.95 threshold")
        print("Consider training CNN if Random Forest performance is insufficient")
    else:
        print(f"\nSUCCESS: Validation F1 score {val_score:.4f} >= 0.95 threshold")

if __name__ == '__main__':
    main()
