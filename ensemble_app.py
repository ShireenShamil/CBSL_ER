"""Ensemble dashboard: bagging vs boosting vs voting vs stacking on LKR exchange-rate returns.
Run:  streamlit run ensemble_app.py
Keep fx_models.py, ensemble_models.py and lkr_fx_clean.csv in the same folder.
"""
import pandas as pd
import streamlit as st
import fx_models as fm
import ensemble_models as em

st.set_page_config(page_title="Ensemble Dashboard", layout="wide")
st.title("Ensemble methods dashboard: LKR exchange rate")
st.caption("Predicting the next-day % change of the mid rate. Bagging builds trees in parallel and averages them; "
           "boosting builds trees one after another, each fixing the last one's errors.")

# ---------------- sidebar ----------------
sb = st.sidebar
up = sb.file_uploader("Cleaned CSV (optional)", type="csv")
df = pd.read_csv(up, parse_dates=["date"]) if up else fm.load("lkr_fx_clean.csv")
cur = sb.selectbox("Currency (vs LKR)", ["USD", "GBP", "EUR", "JPY", "AUD", "CHF"])
X = fm.make_features(df, cur)
train_end = sb.date_input("Train until", value=pd.Timestamp("2023-12-31").date(),
                          min_value=pd.Timestamp("2010-01-01").date(), max_value=X["date"].max().date())

sb.subheader("Shared settings")
params = {"n_estimators": sb.slider("Trees / boosting rounds", 50, 500, 200, 50),
          "max_depth": sb.slider("Max depth", 2, 10, 4),
          "learning_rate": sb.select_slider("Learning rate (boosting)", [0.01, 0.03, 0.05, 0.1, 0.2, 0.3], 0.05),
          "min_samples_leaf": sb.slider("Min samples per leaf", 1, 100, 20),
          "seed": sb.number_input("Random seed", 0, 9999, 42)}

sb.subheader("Models")
avail = em.available_models()
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
        st.session_state["res"] = em.run_ensemble_experiment(X, str(train_end), params, selected, vote, stack)
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
