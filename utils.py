import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from config import (
    DATA_PATH,
    LOG_LEVEL,
    MODEL_DIR,
    METRICS_PATH,
    TARGET_COLUMN,
    TEXT_COLUMNS,
)


def setup_logging() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    logger = logging.getLogger("utils.load_dataset")
    if not os.path.exists(path):
        logger.error("Dataset not found at %s", path)
        raise FileNotFoundError(f"Dataset not found at {path}")
    try:
        df = pd.read_csv(path)
        logger.info("Loaded dataset with shape %s", df.shape)
        return df
    except Exception as e:
        logger.exception("Failed to load dataset: %s", e)
        raise


def detect_schema(df: pd.DataFrame) -> Dict[str, List[str]]:
    logger = logging.getLogger("utils.detect_schema")
    logger.info("Detecting schema from dataframe dtypes:\n%s", df.dtypes)

    cols = df.columns.tolist()
    target_col = TARGET_COLUMN if TARGET_COLUMN in cols else None

    # ID-like columns: unique values ~ len(df)
    id_like_cols: List[str] = []
    high_null_cols: List[str] = []
    constant_cols: List[str] = []

    for col in cols:
        col_series = df[col]
        if col_series.nunique(dropna=True) >= 0.9 * len(df):
            id_like_cols.append(col)
        if col_series.isna().mean() > 0.5:
            high_null_cols.append(col)
        if col_series.nunique(dropna=True) <= 1:
            constant_cols.append(col)

    # Drop unwanted columns
    drop_cols = set(id_like_cols + high_null_cols + constant_cols)
    if target_col in drop_cols:
        drop_cols.remove(target_col)

    if drop_cols:
        logger.info("Dropping columns: %s", list(drop_cols))
        df = df.drop(columns=list(drop_cols))

    # Recompute columns after drops
    cols = df.columns.tolist()
    target_col = TARGET_COLUMN if TARGET_COLUMN in cols else target_col

    # Identify numeric / categorical / text candidates
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    text_cols: List[str] = []

    for col in cols:
        if col == target_col:
            continue
        dtype = df[col].dtype
        if np.issubdtype(dtype, np.number):
            numeric_cols.append(col)
        else:
            # decide between categorical vs text by cardinality and average length
            nunique = df[col].nunique(dropna=True)
            avg_len = (
                df[col].astype(str).str.len().mean()
                if not np.issubdtype(dtype, np.number)
                else 0
            )
            if nunique <= 30 and avg_len < 30:
                categorical_cols.append(col)
            else:
                text_cols.append(col)

    # Enforce configured TEXT_COLUMNS if present in df
    configured_text_cols = [c for c in TEXT_COLUMNS if c in df.columns and c != target_col]
    for c in configured_text_cols:
        if c in numeric_cols:
            numeric_cols.remove(c)
        if c in categorical_cols:
            categorical_cols.remove(c)
        if c not in text_cols:
            text_cols.append(c)

    logger.info(
        "Schema detection result - target: %s, numeric: %s, categorical: %s, text: %s",
        target_col,
        numeric_cols,
        categorical_cols,
        text_cols,
    )

    return {
        "target": target_col,
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "text": text_cols,
        "dropped": list(drop_cols),
        "id_like": id_like_cols,
        "high_null": high_null_cols,
        "constant": constant_cols,
    }


def _build_text_transformer(df: pd.DataFrame, text_cols: List[str]) -> BaseEstimator:
    logger = logging.getLogger("utils._build_text_transformer")
    if not text_cols:
        return "drop"

    sample_text = df[text_cols].astype(str).agg(" ".join)
    contains_upper = any(ch.isupper() for ch in sample_text)
    contains_punct = any(not ch.isalnum() and not ch.isspace() for ch in sample_text)
    use_stopwords = len(df) > 500

    logger.info(
        "Text processing decisions - lowercase: %s, remove_punctuation: %s, remove_stopwords: %s",
        contains_upper,
        contains_punct,
        use_stopwords,
    )

    # TfidfVectorizer lowercases and strips accents by default;
    # we adapt lowercase option only if needed.
    vectorizer = TfidfVectorizer(
        lowercase=contains_upper,
        stop_words="english" if use_stopwords else None,
    )

    # For multiple text columns, concatenate into a single feature
    return Pipeline(
        steps=[
            (
                "combine_text",
                FunctionTransformerForText(columns=text_cols),
            ),
            ("tfidf", vectorizer),
        ]
    )


class FunctionTransformerForText(BaseEstimator):
    def __init__(self, columns: List[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        return self

    def transform(self, X: pd.DataFrame):
        text_series = X[self.columns].astype(str).agg(" ".join, axis=1)
        return text_series.values


def build_preprocessing_pipeline(
    df: pd.DataFrame, text_cols: List[str], target_col: Optional[str]
) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    logger = logging.getLogger("utils.build_preprocessing_pipeline")
    schema = detect_schema(df)

    if target_col is None:
        target_col = schema["target"]

    if target_col is None or target_col not in df.columns:
        raise ValueError("Target column could not be detected in dataframe.")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    numeric_cols = schema["numeric"]
    categorical_cols = schema["categorical"]
    detected_text_cols = schema["text"]

    if not text_cols:
        text_cols = detected_text_cols

    logger.info(
        "Building preprocessing pipeline with numeric=%s, categorical=%s, text=%s",
        numeric_cols,
        categorical_cols,
        text_cols,
    )

    transformers = []

    if numeric_cols:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_cols))
        logger.info("Numeric pipeline: median imputation for columns %s", numeric_cols)

    if categorical_cols:
        low_cardinality = [c for c in categorical_cols if df[c].nunique(dropna=True) <= 30]
        high_cardinality = [c for c in categorical_cols if c not in low_cardinality]

        if low_cardinality:
            cat_low_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore", sparse=True),
                    ),
                ]
            )
            transformers.append(("cat_low", cat_low_pipeline, low_cardinality))
            logger.info("OneHotEncoder for low-cardinality categorical columns %s", low_cardinality)

        if high_cardinality:
            cat_high_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "ordinal",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value", unknown_value=-1
                        ),
                    ),
                ]
            )
            transformers.append(("cat_high", cat_high_pipeline, high_cardinality))
            logger.info("OrdinalEncoder for high-cardinality categorical columns %s", high_cardinality)

    if text_cols:
        text_pipeline = _build_text_transformer(df, text_cols)
        transformers.append(("text", text_pipeline, text_cols))
        logger.info("Text transformer added for columns %s", text_cols)

    if not transformers:
        raise ValueError("No valid features found for preprocessing.")

    preprocessor = ColumnTransformer(transformers=transformers)

    logger.info("Fitting preprocessing pipeline...")
    preprocessor.fit(X, y)
    logger.info("Preprocessing pipeline fitted successfully.")

    return X, y, preprocessor


def save_metrics(metrics: Dict) -> None:
    try:
        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    except Exception as e:
        logging.getLogger("utils.save_metrics").exception("Failed to save metrics: %s", e)


def load_metrics() -> Optional[Dict]:
    if not os.path.exists(METRICS_PATH):
        return None
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logging.getLogger("utils.load_metrics").exception("Failed to load metrics")
        return None
