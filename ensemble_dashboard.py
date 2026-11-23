"""Ensemble dashboard (single file): bagging vs boosting vs voting vs stacking on LKR exchange-rate returns.
Needs only lkr_fx_clean.csv in the same folder.   Run:  streamlit run ensemble_dashboard.py
"""
import warnings
warnings.filterwarnings("ignore")
import streamlit as st

# ===== data, features, metrics =====
import numpy as np
import pandas as pd


CROSS = ["GBP", "EUR", "JPY"]


def load(path):
    return pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)


def make_features(df, cur="USD"):
    mid = (df[f"{cur}_buy"] + df[f"{cur}_sell"]) / 2
    lp = np.log(mid)
    r = lp.diff() * 100                                   # % log return, known at close of day t
    X = pd.DataFrame({"date": df["date"], "mid": mid, "ret_0": r})
    for k in (1, 2, 3, 4, 9):
        X[f"ret_{k}"] = r.shift(k)
    for w in (5, 20):
        X[f"ret_mean_{w}"] = r.rolling(w).mean()
        X[f"ret_std_{w}"] = r.rolling(w).std()
    for w in (20, 60):
        X[f"dev_ma{w}"] = (lp - np.log(mid.rolling(w).mean())) * 100
    X["spread_pct"] = (df[f"{cur}_sell"] - df[f"{cur}_buy"]) / mid * 100
    X["dow"] = df["date"].dt.dayofweek
    X["month"] = df["date"].dt.month
    for c in CROSS:                                       # cross-rate move vs the base currency, same day
        if c != cur and f"{c}_buy" in df:
            cm = (df[f"{c}_buy"] + df[f"{c}_sell"]) / 2
            X[f"x_{c}_0"] = (np.log(cm).diff() * 100) - r
    X["target"] = r.shift(-1)                             # next-day return (what we predict)
    return X.dropna().reset_index(drop=True)


def _metrics(name, y, p, px):
    err = y - p
    lvl_true, lvl_pred = px * np.exp(y / 100), px * np.exp(p / 100)
    nz = np.isfinite(p) & (p != 0) & (y != 0)
    return {
        "model": name,
        "RMSE (% return)": float(np.sqrt(np.mean(err ** 2))),
        "MAE (% return)": float(np.mean(np.abs(err))),
        "Direction acc.": float(np.mean(np.sign(p[nz]) == np.sign(y[nz]))) if nz.any() else np.nan,
        "RMSE (Rs level)": float(np.sqrt(np.mean((lvl_true - lvl_pred) ** 2))),
        "MAE (Rs level)": float(np.mean(np.abs(lvl_true - lvl_pred))),
    }

# ===== ensemble models =====
import time
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor,
                              AdaBoostRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor)
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor


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
        row = _metrics(k, yte, v, px_te)
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


# ===== dashboard =====

st.set_page_config(page_title="Ensemble Dashboard", layout="wide")
st.title("Ensemble methods dashboard: LKR exchange rate")
st.caption("Predicting the next-day % change of the mid rate. Bagging builds trees in parallel and averages them; "
           "boosting builds trees one after another, each fixing the last one's errors.")

# ---------------- sidebar ----------------
sb = st.sidebar
up = sb.file_uploader("Cleaned CSV (optional)", type="csv")
df = pd.read_csv(up, parse_dates=["date"]) if up else load("lkr_fx_clean.csv")
cur = sb.selectbox("Currency (vs LKR)", ["USD", "GBP", "EUR", "JPY", "AUD", "CHF"])
X = make_features(df, cur)
train_end = sb.date_input("Train until", value=pd.Timestamp("2023-12-31").date(),
                          min_value=pd.Timestamp("2010-01-01").date(), max_value=X["date"].max().date())

sb.subheader("Shared settings")
params = {"n_estimators": sb.slider("Trees / boosting rounds", 50, 500, 200, 50),
          "max_depth": sb.slider("Max depth", 2, 10, 4),
          "learning_rate": sb.select_slider("Learning rate (boosting)", [0.01, 0.03, 0.05, 0.1, 0.2, 0.3], 0.05),
          "min_samples_leaf": sb.slider("Min samples per leaf", 1, 100, 20),
          "seed": sb.number_input("Random seed", 0, 9999, 42)}

sb.subheader("Models")
avail = available_models()
default_sel = [m for m in ["Random Forest", "Extra Trees", "Gradient Boosting", "XGBoost", "HistGradientBoosting"] if m in avail]
selected = sb.multiselect("Base models", avail, default=default_sel)
vote = sb.multiselect("Voting ensemble members (average)", selected, default=selected)
stack = sb.multiselect("Stacking ensemble members", selected, default=selected,
                       help="Slow: each member is retrained 5 times on expanding time windows.")
missing = [m for m in ["XGBoost", "LightGBM", "CatBoost"] if m not in avail]
if missing:
    sb.info("Not installed (optional): " + ", ".join(missing) + ". Install with pip to add them.")
run = sb.button("Train / update", type="primary")

if not selected:
    st.info("Pick at least one base model in the sidebar.")
    st.stop()

key = (cur, str(train_end), tuple(sorted(params.items())), tuple(selected), tuple(vote), tuple(stack))
if run or st.session_state.get("key") is None:
    with st.spinner("Training models..."):
        st.session_state["res"] = run_ensemble_experiment(X, str(train_end), params, selected, vote, stack)
        st.session_state["key"] = key
elif st.session_state["key"] != key:
    st.warning("Settings changed. Click **Train / update** in the sidebar to refresh the results.")
res = st.session_state["res"]

c1, c2, c3 = st.columns(3)
c1.metric("Data range", f"{X['date'].min().date()} to {X['date'].max().date()}")
c2.metric("Train rows", res["n_train"])
c3.metric("Test rows", res["n_test"])

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Compare", "Predictions & error", "Feature importance",
                                        "Stacking weights", "Learn: bagging vs boosting"])

with tab1:
    t = res["table"]
    st.caption("Lower RMSE / MAE is better. The naive row (tomorrow = today) is the baseline: a model that does not "
               "beat it adds nothing. 'RMSE vs naive' is the % improvement over that baseline.")
    show = ["Family", "RMSE (% return)", "MAE (% return)", "Direction acc.", "RMSE vs naive (%)", "Train time (s)"]
    fmt = {"RMSE (% return)": "{:.4f}", "MAE (% return)": "{:.4f}", "Direction acc.": "{:.1%}",
           "RMSE vs naive (%)": "{:+.1f}", "Train time (s)": "{:.1f}"}
    st.dataframe(t[show].style.format(fmt, na_rep="-").highlight_min(subset=["RMSE (% return)", "MAE (% return)"],
                                                                      color="#cfe8cf"))
    best = t["RMSE (% return)"].idxmin()
    st.success(f"Lowest RMSE: **{best}** ({t.loc[best, 'RMSE (% return)']:.4f}). "
               "Differences between models are small, so check the result across seeds and split dates "
               "before calling a winner.")
    st.bar_chart(t["RMSE (% return)"].sort_values())
    fam = t.drop(index="Naive (tomorrow = today)").groupby("Family")["RMSE (% return)"].mean()
    st.write("**Average RMSE by family**")
    st.bar_chart(fam)

with tab2:
    tt = res["test"].set_index("date")
    cols = [c for c in tt.columns if c not in ("mid", "target")]
    pick = st.multiselect("Models to plot", cols, default=[c for c in cols if c != "Naive (tomorrow = today)"][:3])
    n = st.slider("Last N test days", 30, len(tt), min(150, len(tt)))
    if pick:
        st.line_chart(tt[["target"] + pick].tail(n).rename(columns={"target": "actual"}))
    st.write("**Cumulative absolute error** (lower is better)")
    st.line_chart(pd.DataFrame({c: (tt["target"] - tt[c]).abs().cumsum() for c in cols}))

with tab3:
    imp = res["importances"]
    if imp.empty:
        st.info("No tree-based models selected.")
    else:
        which = st.multiselect("Models", list(imp.columns), default=list(imp.columns)[:3])
        if which:
            st.bar_chart(imp[which].sort_values(which[0]))

with tab4:
    w = res["weights"]
    if w is None:
        st.info("Select at least 2 stacking members in the sidebar.")
    else:
        st.write("Ridge meta-model weights: how much each base model's prediction counts in the stacked forecast.")
        st.bar_chart(w)
        st.dataframe(w.to_frame().style.format("{:+.3f}"))

with tab5:
    st.markdown("""
| Method | How it works | Strengths | Limitations |
|---|---|---|---|
| **Bagging** (Random Forest, Extra Trees) | Many trees trained independently on random samples; predictions averaged | Reduces variance, hard to overfit, little tuning needed | Cannot fix bias; averages out real but weak signals |
| **Boosting** (AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost) | Trees built one after another, each correcting the previous errors | Often the most accurate on tabular data | More tuning needed; can overfit noisy data |
| **Voting** | Simple average of chosen models | Easy, robust, no extra training | Weak models drag the average down |
| **Stacking** | A meta-model learns how to weight the base models | Can beat every single model | Slow, can overfit, needs time-aware validation |

**Tabular vs image/text data:** tree ensembles are usually the strongest choice for tabular data like this exchange-rate table
(mixed numeric features, modest rows, little preprocessing). For images and text, deep learning (CNNs, transformers) usually wins
because those models learn features from raw pixels or words, which trees cannot do.

**Why returns, not rates?** Trees cannot predict outside the range seen in training. The rupee jumped from about 200 to 360 in 2022,
so models predict the daily % change instead of the raw rate.
""")
