import logging
import os
from typing import Dict, Tuple

import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    DATA_PATH,
    MODEL_DIR,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    TEXT_COLUMNS,
    TARGET_COLUMN,
)
from utils import build_preprocessing_pipeline, load_dataset, save_metrics, setup_logging


def _build_models(preprocessor) -> Dict[str, Pipeline]:
    models = {}

    # Logistic Regression with scaling (only here)
    log_reg_clf = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    models["logistic_regression"] = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", log_reg_clf),
        ]
    )

    # Random Forest without scaling
    rf_clf = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    models["random_forest"] = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", rf_clf),
        ]
    )

    # XGBoost without scaling
    xgb_clf = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    models["xgboost"] = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", xgb_clf),
        ]
    )

    return models


def train_and_select_model() -> Tuple[str, Pipeline, Dict]:
    logger = logging.getLogger("train.train_and_select_model")

    df = load_dataset(DATA_PATH)
    X, y, preprocessor = build_preprocessing_pipeline(df, TEXT_COLUMNS, TARGET_COLUMN)

    # Encode target if not numeric
    if not np.issubdtype(y.dtype, np.number):
        classes, y_encoded = np.unique(y, return_inverse=True)
        y = y_encoded
        target_mapping = {int(i): str(c) for i, c in enumerate(classes)}
    else:
        target_mapping = None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    models = _build_models(preprocessor)

    best_model_name = None
    best_model = None
    best_accuracy = -1.0
    all_metrics: Dict[str, Dict] = {}

    for name, model in models.items():
        logger.info("Training model: %s", name)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        logger.info("Model %s accuracy: %.4f", name, acc)

        cls_report = classification_report(
            y_test, preds, output_dict=True, zero_division=0
        )
        cm = confusion_matrix(y_test, preds).tolist()

        all_metrics[name] = {
            "accuracy": acc,
            "classification_report": cls_report,
            "confusion_matrix": cm,
            "target_mapping": target_mapping,
        }

        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            best_model = model

    if best_model is None or best_model_name is None:
        raise RuntimeError("No model was successfully trained.")

    logger.info("Best model selected: %s with accuracy %.4f", best_model_name, best_accuracy)

    # Compute feature importance if tree-based
    feature_importance = None
    clf = best_model.named_steps.get("clf")
    if hasattr(clf, "feature_importances_"):
        try:
            importance = clf.feature_importances_.tolist()
            feature_importance = {"feature_importances": importance}
            all_metrics[best_model_name]["feature_importance"] = feature_importance
        except Exception:
            logger.exception("Failed to extract feature importances.")

    overall_metrics = {
        "best_model": best_model_name,
        "models": all_metrics,
    }

    save_metrics(overall_metrics)

    # Save both the preprocessor and the best model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(best_model, MODEL_PATH)

    logger.info("Saved preprocessor to %s and model to %s", PREPROCESSOR_PATH, MODEL_PATH)

    return best_model_name, best_model, overall_metrics


def main() -> None:
    setup_logging()
    logger = logging.getLogger("train.main")
    try:
        logger.info("Starting training pipeline...")
        train_and_select_model()
        logger.info("Training pipeline completed successfully.")
    except Exception as e:
        logger.exception("Training pipeline failed: %s", e)
        raise


if __name__ == "__main__":
    main()

