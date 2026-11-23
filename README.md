# Ensemble Methods for Sri Lankan Rupee (LKR) Exchange Rate Forecasting

Compares **bagging**, **boosting**, **voting** and **stacking** ensembles for predicting the next-day movement of the USD/LKR exchange rate, using 20+ years of Central Bank of Sri Lanka (CBSL) data. Includes a Jupyter notebook and an interactive Streamlit dashboard.

**Main finding:** boosting models beat the "tomorrow = today" baseline by about 4 to 8%, bagging barely moved, and stacking did worse than the baseline. The gaps are small, so the result should be read as modest, not decisive.

## Files

| File | What it is |
|---|---|
| `lkr_fx_clean.csv` | Cleaned daily buying/selling rates, 3 Jan 2005 to 30 Oct 2026 |
| `lkr_ensemble_notebook.ipynb` | Self-contained notebook: trains and compares all ensembles, with charts and stability checks |
| `ensemble_dashboard.py` | Self-contained Streamlit dashboard (single file) |
| `requirements.txt` | Python dependencies |

Optional multi-file versions (same functionality, split into modules): `ensemble_app.py` + `ensemble_models.py` + `fx_models.py`. The earlier `app.py` and `lkr_fx_comparison*.ipynb` cover a first RF vs XGBoost vs ARIMA comparison.

The notebook and the dashboard each need only `lkr_fx_clean.csv` in the same folder.

## Quick start

```bash
pip install -r requirements.txt
pip install notebook            # for the notebook
pip install lightgbm catboost   # optional extra boosting models
```

**Notebook**
```bash
python -m jupyter notebook
```
Open `lkr_ensemble_notebook.ipynb` and choose Kernel > Restart & Run All. The stacking and stability cells take a few minutes.

**Dashboard**
```bash
streamlit run ensemble_dashboard.py
```
Press Enter if Streamlit asks for an email, then open `http://localhost:8501` if the browser does not open. Click **Train / update** in the sidebar after changing settings.

## Data

- **Source:** CBSL, buying and selling exchange rates of commercial banks (Telegraphic Transfer rates).
- **Currencies:** USD (default), GBP, EUR, JPY, AUD, CHF are selectable in the dashboard.
- **Cleaning:** removed monthly "Average" summary rows and footnotes, sorted by date, and dropped one erroneous duplicate row each for 2015-10-06 and 2018-10-01.
- **Mid rate:** (buying + selling) / 2.
- **Usable range:** modelling starts in April 2005, because the first rows are used up by the rolling-window features (up to 60 days).
- This is the TT buying/selling series, not CBSL's indicative spot rate.

## Method

**Target:** next-day log return of the mid rate, in percent. Predicting returns instead of raw rates matters because tree models cannot predict outside the range they saw in training, and the rupee jumped from about 200 to 360 in 2022.

**Features** (all known at the close of day *t*): returns today and 1, 2, 3, 4 and 9 days ago; 5-day and 20-day rolling mean and standard deviation of returns; distance from the 20-day and 60-day moving averages; buy/sell spread; day of week; month; cross-currency moves (GBP, EUR, JPY).

**Split:** strictly time-ordered, no shuffling. Default: train up to 31 Dec 2023 (4,519 rows), test from 2024 onward (684 rows).

**Models**

| Family | Models |
|---|---|
| Baseline | Naive (tomorrow = today) |
| Reference | Single decision tree |
| Bagging | Random Forest, Extra Trees, Bagging (trees) |
| Boosting | AdaBoost, Gradient Boosting, HistGradientBoosting, XGBoost*, LightGBM*, CatBoost* |
| Combined | Voting (average of chosen models), Stacking (Ridge meta-model) |

\* used only if the package is installed.

Stacking trains the base models on expanding time windows (`TimeSeriesSplit`, 5 splits) and fits a Ridge model on their out-of-fold predictions, so no future data leaks in. It is slow because every member is retrained several times.

**Metrics:** RMSE and MAE on the % return, direction accuracy, improvement over the naive baseline, and training time.

## Dashboard tabs

1. **Compare:** metrics table, RMSE by model, average RMSE by family
2. **Predictions & error:** predicted vs actual returns, cumulative absolute error
3. **Feature importance:** for tree-based models
4. **Stacking weights:** how much each base model counts in the stacked forecast
5. **Learn:** bagging vs boosting vs voting vs stacking, and tabular vs image/text data

## Results

Test period 2024 to Oct 2026 (684 trading days). Settings: 200 trees/rounds, max depth 4, learning rate 0.05, min samples per leaf 20, seed 42.

| Model | Family | RMSE (% return) | MAE (% return) | Direction acc. | RMSE vs naive |
|---|---|---|---|---|---|
| Naive (tomorrow = today) | baseline | 0.3446 | 0.1533 | n/a | 0.0% |
| Random Forest | bagging | 0.3382 | 0.1494 | 59.5% | +1.9% |
| Extra Trees | bagging | 0.3446 | 0.1508 | 56.3% | 0.0% |
| **Gradient Boosting** | boosting | **0.3165** | 0.1497 | 62.2% | **+8.2%** |
| XGBoost | boosting | 0.3245 | **0.1444** | 65.8% | +5.8% |
| HistGradientBoosting | boosting | 0.3296 | 0.1453 | **66.4%** | +4.3% |
| Voting (average) | combined | 0.3258 | 0.1447 | 64.4% | +5.5% |
| Stacking (Ridge) | combined | 0.3537 | 0.1580 | 52.8% | -2.7% |

Average RMSE by family: boosting about 0.324, combined about 0.340, bagging about 0.341.

Gradient Boosting has the lowest RMSE, XGBoost the lowest MAE, and HistGradientBoosting the best direction accuracy.

## Why these results? (likely explanations)

These are explanations based on how the models work, not proven causes.

- **Small gains overall:** daily exchange-rate moves are mostly noise with a very small signal. "Tomorrow = today" is already close to the best guess on most days, and the error is dominated by rare large moves.
- **Boosting beat bagging:** Random Forest and Extra Trees average many trees, which pulls predictions toward zero, so they end up near the baseline. Boosting builds shallow trees one at a time with a small learning rate, so it can pick up a small, consistent pattern.
- **Different winners by metric:** RMSE punishes large misses, so Gradient Boosting probably handles big-move days best, while XGBoost has the smallest typical error (MAE).
- **Direction accuracy of 62 to 66% with small error gains:** the dominant features suggest mean reversion toward a recent average, which helps with direction but barely with the size of moves. The series is also an average of bank quotes, which smooths it.
- **Not a trading signal:** the buy-sell spread is a few rupees, far bigger than a typical daily move.
- **Stacking was worse than the baseline:** all base models are trees on the same features, so their predictions are similar and there is little diversity to exploit. The meta-model also learns its weights from earlier, calmer years, which may not carry over to the test period.
- **Voting landed in the middle:** an average of good and weak models gives middle performance.

## Checking stability

The gaps between models are small, so check them before naming a winner:
- Notebook section 9 re-runs the comparison across random seeds and four training end dates.
- In the dashboard, change **Train until** (for example to 2021-12-31, so the test period includes the 2022 currency crisis) and click **Train / update**.
- In a preliminary test, boosting lost its edge when the test period included 2022, consistent with a structural break (the rupee moved from a tightly managed rate to a float). Verify this on your own run before relying on it.

## Interpretation

- Improvements over the naive baseline are modest. Exchange rates are hard to predict, so ensembles help a little, not a lot.
- Rupee-level errors look tiny for every model because tomorrow's rate is always close to today's, so compare models using the % return errors.
- Tree ensembles suit tabular data like this. For images and text, deep learning usually performs better because it learns features from raw pixels or words.

## Limitations

- One time-ordered train/test split in the main comparison (stability is checked separately across split dates and seeds).
- Limited hyperparameter tuning (shared sliders only).
- Data ends 30 Oct 2026, the latest CBSL file available at download.
- Results do not constitute trading advice.

## Possible extensions

- Walk-forward validation
- Add oil prices, T-bill yields or the stock index as features
- Tune LightGBM and CatBoost
- Run pre-crisis and post-crisis periods separately
- Add an "error by year" chart to show the effect of the 2022 crisis directly
