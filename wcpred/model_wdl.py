"""wcpred.model_wdl — XGBoost Win/Draw/Loss model: split, train, evaluate, predict."""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss

from wcpred.features import FEATURES, build_match_row

TRAIN_START = "2006-01-01"
VAL_START = "2023-01-01"
MATCH_WEIGHT = 4          # FIFA World Cup
MATCH_NEUTRAL = True      # 2026 group games at neutral US/CA/MX venues for these teams


# ── model ───────────────────────────────────────────────────────────────────────
def split_by_date(ds, train_start, val_start, cutoff):
    train = ds[(ds["date"] >= pd.Timestamp(train_start)) & (ds["date"] < pd.Timestamp(val_start))].copy()
    val = ds[(ds["date"] >= pd.Timestamp(val_start)) & (ds["date"] < pd.Timestamp(cutoff))].copy()
    return train, val


def train_model(train, val):
    X_train, y_train = train[FEATURES].astype(float), train["label"].astype(int)
    X_val, y_val = val[FEATURES].astype(float), val["label"].astype(int)
    model = xgb.XGBClassifier(
        objective="multi:softprob", num_class=3, n_estimators=600,
        learning_rate=0.05, max_depth=5, subsample=0.85, colsample_bytree=0.85,
        reg_lambda=1.0, eval_metric="mlogloss", early_stopping_rounds=50,
        tree_method="hist", n_jobs=-1, random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model, X_val, y_val


def evaluate(model, X_val, y_val):
    proba = model.predict_proba(X_val)
    pred = proba.argmax(axis=1)
    base = np.tile(np.bincount(y_val, minlength=3) / len(y_val), (len(y_val), 1))
    print(f"  Validation accuracy : {accuracy_score(y_val, pred):.3f}")
    print(f"  Validation log-loss : {log_loss(y_val, proba):.3f}  (baseline {log_loss(y_val, base, labels=[0,1,2]):.3f})")


def predict_symmetric(model, long, final_elo, a, b, asof, neutral, weight):
    p_ab = model.predict_proba(build_match_row(long, final_elo, a, b, neutral, weight, asof))[0]
    p_ba = model.predict_proba(build_match_row(long, final_elo, b, a, neutral, weight, asof))[0]
    p_a = (p_ab[0] + p_ba[2]) / 2.0
    p_d = (p_ab[1] + p_ba[1]) / 2.0
    p_b = (p_ab[2] + p_ba[0]) / 2.0
    tot = p_a + p_d + p_b
    return p_a / tot, p_d / tot, p_b / tot
