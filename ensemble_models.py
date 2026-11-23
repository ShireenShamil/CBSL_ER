"""Ensemble methods for next-day LKR return prediction.

Families compared
  bagging   : Random Forest, Extra Trees, Bagging (trees)
  boosting  : AdaBoost, Gradient Boosting, HistGradientBoosting, XGBoost*, LightGBM*, CatBoost*
  combined  : Voting (average of chosen models) and Stacking (Ridge meta-model)
  reference : a single Decision Tree (shows why ensembles help)
* optional: used only if the package is installed.
Everything is time-ordered: stacking uses an expanding-window TimeSeriesSplit, never shuffled folds.
"""
import time
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor,
                              AdaBoostRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor)
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor

import fx_models as fm

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None
try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None
try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

FAMILY = {"Decision Tree (single)": "reference",
          "Random Forest": "bagging", "Extra Trees": "bagging", "Bagging (trees)": "bagging",
          "AdaBoost": "boosting", "Gradient Boosting": "boosting", "HistGradientBoosting": "boosting",
          "XGBoost": "boosting", "LightGBM": "boosting", "CatBoost": "boosting"}

DEFAULTS = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "min_samples_leaf": 20, "seed": 42}


def build_models(p=None):
    p = {**DEFAULTS, **(p or {})}
    n, d, lr, leaf, s = p["n_estimators"], p["max_depth"], p["learning_rate"], p["min_samples_leaf"], p["seed"]
    m = {
        "Decision Tree (single)": DecisionTreeRegressor(max_depth=d, min_samples_leaf=leaf, random_state=s),
        "Random Forest": RandomForestRegressor(n_estimators=n, max_depth=d, min_samples_leaf=leaf, max_features=0.5,
                                                n_jobs=-1, random_state=s),
        "Extra Trees": ExtraTreesRegressor(n_estimators=n, max_depth=d, min_samples_leaf=leaf, max_features=0.5,
                                            n_jobs=-1, random_state=s),
        "Bagging (trees)": BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=d, min_samples_leaf=leaf),
                                            n_estimators=n, n_jobs=-1, random_state=s),
        "AdaBoost": AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=min(d, 3)),
                                      n_estimators=n, learning_rate=lr, random_state=s),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=n, max_depth=min(d, 5), learning_rate=lr,
                                                       subsample=0.8, random_state=s),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=n, max_depth=d, learning_rate=lr, random_state=s),
    }
    if XGBRegressor is not None:
        m["XGBoost"] = XGBRegressor(n_estimators=n, max_depth=d, learning_rate=lr, subsample=0.8,
                                    colsample_bytree=0.8, random_state=s, n_jobs=-1)
    if LGBMRegressor is not None:
        m["LightGBM"] = LGBMRegressor(n_estimators=n, max_depth=d, learning_rate=lr, subsample=0.8,
                                      subsample_freq=1, colsample_bytree=0.8, random_state=s, verbose=-1)
    if CatBoostRegressor is not None:
        m["CatBoost"] = CatBoostRegressor(iterations=n, depth=min(d, 8), learning_rate=lr, random_seed=s, verbose=0)
    return m


def available_models():
    return list(build_models().keys())


def run_ensemble_experiment(X, train_end="2023-12-31", params=None, selected=None,
                            vote_members=None, stack_members=None, n_splits=5):
    models = build_models(params)
    selected = [m for m in (selected or list(models)) if m in models]
    vote_members = [m for m in (vote_members or []) if m in selected]
    stack_members = [m for m in (stack_members or []) if m in selected]

    feats = [c for c in X.columns if c not in ("date", "mid", "target")]
    tr = (X["date"] <= pd.Timestamp(train_end)).values
    Xtr, Xte = X.loc[tr, feats].reset_index(drop=True), X.loc[~tr, feats].reset_index(drop=True)
    ytr, yte = X.loc[tr, "target"].values, X.loc[~tr, "target"].values
    px_te = X.loc[~tr, "mid"].values

    preds = {"Naive (tomorrow = today)": np.zeros(len(yte))}
    secs, imps = {}, {}
    for name in selected:
        t0 = time.perf_counter()
        est = models[name].fit(Xtr, ytr)
        secs[name] = time.perf_counter() - t0
        preds[name] = np.asarray(est.predict(Xte), dtype=float)
        if hasattr(est, "feature_importances_"):
            imps[name] = pd.Series(est.feature_importances_, feats)

    if len(vote_members) >= 2:
        t0 = time.perf_counter()
        preds["Voting ensemble (average)"] = np.mean([preds[m] for m in vote_members], axis=0)
        secs["Voting ensemble (average)"] = sum(secs[m] for m in vote_members) + (time.perf_counter() - t0)

    weights = None
    if len(stack_members) >= 2:
        t0 = time.perf_counter()
        oof = np.full((len(ytr), len(stack_members)), np.nan)
        for tr_i, va_i in TimeSeriesSplit(n_splits=n_splits).split(Xtr):      # expanding window, no leakage
            for j, name in enumerate(stack_members):
                est = clone(models[name]).fit(Xtr.iloc[tr_i], ytr[tr_i])
                oof[va_i, j] = est.predict(Xtr.iloc[va_i])
        ok = ~np.isnan(oof).any(axis=1)
        meta = Ridge(alpha=1.0).fit(oof[ok], ytr[ok])
        test_stack = np.column_stack([preds[m] for m in stack_members])
        preds["Stacking ensemble (Ridge)"] = meta.predict(test_stack)
        secs["Stacking ensemble (Ridge)"] = sum(secs[m] for m in stack_members) + (time.perf_counter() - t0)
        weights = pd.Series(meta.coef_, stack_members, name="Ridge weight")

    rows = []
    for k, v in preds.items():
        row = fm._metrics(k, yte, v, px_te)
        row["Family"] = ("baseline" if k.startswith("Naive") else
                         "combined" if "ensemble" in k else FAMILY.get(k, "other"))
        row["Train time (s)"] = secs.get(k, np.nan)
        rows.append(row)
    table = pd.DataFrame(rows).set_index("model")
    naive = table.loc["Naive (tomorrow = today)", "RMSE (% return)"]
    table["RMSE vs naive (%)"] = (1 - table["RMSE (% return)"] / naive) * 100
    out = X.loc[~tr, ["date", "mid", "target"]].reset_index(drop=True)
    for k, v in preds.items():
        out[k] = v
    return {"table": table, "test": out, "importances": pd.DataFrame(imps), "weights": weights,
            "n_train": int(tr.sum()), "n_test": int((~tr).sum())}
