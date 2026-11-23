# Ensemble Methods for Sri Lankan Rupee (LKR) Exchange Rate Forecasting

Compares **bagging**, **boosting**, **voting** and **stacking** ensembles for predicting the next-day movement of the USD/LKR exchange rate, using 20+ years of Central Bank of Sri Lanka (CBSL) data. Includes a Jupyter notebook and an interactive Streamlit dashboard.

## Files

| File | What it is |
|---|---|
| `lkr_fx_clean.csv` | Cleaned daily buying/selling rates, 3 Jan 2005 to 30 Oct 2026 |
| `lkr_ensemble_notebook.ipynb` | Self-contained notebook: trains and compares all ensembles, with charts and stability checks |
| `ensemble_dashboard.py` | Self-contained Streamlit dashboard (single file) |
| `requirements.txt` | Python dependencies |

Optional multi-file versions (same functionality, split into modules): `ensemble_app.py` + `ensemble_models.py` + `fx_models.py`. The earlier `app.py`, `lkr_fx_comparison*.ipynb` cover a first RF vs XGBoost vs ARIMA comparison.

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
- This is the TT buying/selling series, not CBSL's indicative spot rate.

## Method

**Target:** next-day log return of the mid rate, in percent. Predicting returns instead of raw rates matters because tree models cannot predict outside the range they saw in training, and the rupee jumped from about 200 to 360 in 2022.

**Features** (all known at the close of day *t*): returns today and 1, 2, 3, 4 and 9 days ago; 5-day and 20-day rolling mean and standard deviation of returns; distance from the 20-day and 60-day moving averages; buy/sell spread; day of week; month; cross-currency moves (GBP, EUR, JPY).

**Split:** strictly time-ordered. Default: train up to 31 Dec 2023, test from 2024 onward. No shuffling.

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

Fill in after running the notebook (section 3 and section 9):

| Model | RMSE (% return) | Direction acc. | vs naive |
|---|---|---|---|
| Naive (tomorrow = today) | | n/a | 0% |
| Best bagging model: | | | |
| Best boosting model: | | | |
| Voting ensemble | | | |
| Stacking ensemble | | | |

Observations from a preliminary run (scikit-learn models only, so check them on your own run):
- Boosting models had the lowest average error for the 2023 and 2024 split dates.
- When the test period included the 2022 crisis (training ended 2021), boosting was weaker than the naive baseline.
- Stacking did not beat the best single model.

## Interpretation

- Improvements over the naive baseline are modest. Exchange rates are hard to predict, so ensembles help a little, not a lot.
- Differences between models are small. Check the seed and split-date tables (notebook section 9) before naming a winner.
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
