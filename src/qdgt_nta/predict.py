"""Batch predictions and applicability domain checks."""
from pathlib import Path
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from .features import generate_features

ROOT = Path(os.environ.get("QDGT_DATA_DIR", Path.cwd())).resolve()
CONFIG = {
    "d": ("-logD_XGBOOST.pkl", "-logD_train_features.npy", "Predicted_logD", "in_AD_D", .15, 1),
    "pos": ("logIE_pos_XGBOOST.pkl", "logIE_pos-train_features.npy", "Predicted_logIE_POS", "in_AD_IE(+)", .55, 3),
    "neg": ("logIE_neg_XGBOOST.pkl", "logIE_neg_train_features.npy", "Predicted_logIE_neg", "in_AD_IE(-)", .35, 1),
}


def process_frame(frame, kind, prediction=True, ad=False, similarity=None, count=None, root=ROOT):
    if kind not in CONFIG:
        raise ValueError(f"Unknown model {kind!r}; choose d, pos, or neg")
    if "SMILES" not in frame:
        raise ValueError("Input must contain an SMILES column")
    model_file, train_file, pred_col, ad_col, default_sim, default_count = CONFIG[kind]
    result = frame.copy()
    if prediction:
        result[pred_col] = np.nan
    if ad:
        result[ad_col] = pd.NA
    rows, feature_rows = [], []
    tabular_columns = [c for c in frame if c not in ("Name", "SMILES")]
    for index, row in frame.iterrows():
        try:
            features = generate_features(row["SMILES"], kind)
            if kind != "d":
                features = np.concatenate((features, pd.to_numeric(row[tabular_columns], errors="raise").to_numpy(dtype=float)))
            if np.isinf(features).any():
                raise ValueError("infinite feature")
        except (ValueError, TypeError):
            continue
        rows.append(index)
        feature_rows.append(features)
    if not rows:
        return result
    matrix = np.vstack(feature_rows)
    if prediction:
        model = joblib.load(root / "assets/models" / model_file)
        if matrix.shape[1] != model.n_features_in_:
            raise ValueError(f"Expected {model.n_features_in_} features, got {matrix.shape[1]}; check input columns and dependency versions")
        result.loc[rows, pred_col] = model.predict(matrix)
    if ad:
        train = np.load(root / "assets/train_features" / train_file)
        if train.shape[1] != matrix.shape[1]:
            raise ValueError(f"AD expects {train.shape[1]} features, got {matrix.shape[1]}")
        scaled = StandardScaler().fit_transform(np.nan_to_num(np.vstack((train, matrix))))
        similar = cosine_similarity(scaled[:len(train)], scaled[len(train):])
        result.loc[rows, ad_col] = (similar >= (default_sim if similarity is None else similarity)).sum(axis=0) >= (default_count if count is None else count)
    return result


def process_file(input_path, output_path, kind, **kwargs):
    source = Path(input_path)
    frame = pd.read_csv(source) if source.suffix.lower() == ".csv" else pd.read_excel(source)
    result = process_frame(frame, kind, **kwargs)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".csv":
        result.to_csv(output, index=False)
    else:
        result.to_excel(output, index=False)
    return result
