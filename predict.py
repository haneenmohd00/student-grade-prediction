import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from config import MODEL_PATH, PREPROCESSOR_PATH, TARGET_COLUMN
from train import main as train_main
from utils import load_metrics, setup_logging


logger = logging.getLogger("predict")


def ensure_model() -> Tuple[Pipeline, Dict]:
    setup_logging()
    logger.info("Ensuring model and preprocessor are available.")

    if not (os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH)):
        logger.info("Model or preprocessor not found. Triggering training.")
        train_main()

    try:
        model: Pipeline = joblib.load(MODEL_PATH)
        metrics = load_metrics() or {}
        logger.info("Loaded model from %s", MODEL_PATH)
        return model, metrics
    except Exception as e:
        logger.exception("Failed to load model: %s", e)
        raise


def predict_from_dict(model: Pipeline, data: Dict[str, Any]) -> Dict[str, Any]:
    df = pd.DataFrame([data])

    preds = model.predict(df)

    result: Dict[str, Any] = {}
    metrics = load_metrics() or {}
    best_model_name = metrics.get("best_model")
    target_mapping = None
    if best_model_name:
        target_mapping = (
            metrics.get("models", {})
            .get(best_model_name, {})
            .get("target_mapping")
        )

    if target_mapping:
        # Inverse mapping from encoded index to original label
        try:
            class_index = int(preds[0])
            result["prediction"] = target_mapping.get(str(class_index), str(class_index))
        except Exception:
            result["prediction"] = str(preds[0])
    else:
        result["prediction"] = str(preds[0])

    if hasattr(model.named_steps.get("clf"), "predict_proba"):
        try:
            probas = model.predict_proba(df)[0]
            confidence = float(np.max(probas))
            result["confidence"] = confidence
        except Exception:
            result["confidence"] = None
    else:
        result["confidence"] = None

    return result


def predict_batch(model: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    preds = model.predict(df)
    metrics = load_metrics() or {}
    best_model_name = metrics.get("best_model")
    target_mapping = None
    if best_model_name:
        target_mapping = (
            metrics.get("models", {})
            .get(best_model_name, {})
            .get("target_mapping")
        )

    if target_mapping:
        mapped_preds: List[str] = []
        for p in preds:
            try:
                mapped_preds.append(target_mapping.get(str(int(p)), str(p)))
            except Exception:
                mapped_preds.append(str(p))
        df[TARGET_COLUMN] = mapped_preds
    else:
        df[TARGET_COLUMN] = preds

    return df


if __name__ == "__main__":
    setup_logging()
    try:
        model, _ = ensure_model()
        sample = {}
        print(predict_from_dict(model, sample))
    except Exception as exc:
        logger.exception("Prediction from CLI failed: %s", exc)

