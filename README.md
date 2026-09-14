# Capstone
## Repository structure

```text
Capstone/
├── README.md                                  # This file
└── paper_replication/                         # Replication of OIES Energy Insight 177
    ├── spec1.ipynb                            # Replication specification (equations, data contract)
    ├── data_required/
    │   ├── futures_generic_prices.csv         # Daily generic futures curve (Bloomberg)
    │   ├── contract_meta.csv                  # Contract expiry schedule (Bloomberg)
    │   ├── managed_money.csv                  # Weekly COT Managed Money positions (Bloomberg)
    │   └── sg_cta_indices.csv                 # SG CTA / Trend indices (SocGen API)
    ├── src/paper_replication/
    │   ├── config.py                          # Markets, momentum horizons, ReplicationConfig defaults
    │   ├── data.py                            # Loading, rolling contract, momentum features, weekly panel
    │   ├── model.py                           # PyTorch shared momentum network
    │   └── evaluate.py                        # Rolling out-of-sample evaluation of the network
    ├── notebooks/
    │   ├── linear_regression.ipynb            # Simple linear regression benchmark (paper Tables 1-2)
    │   └── replicate_insight_177.ipynb        # Neural network model
    └── outputs/
        ├── linear_regression/                 # Predictions, R² and accuracy tables, comparison with paper, figures/
        └── neural_network/                    # Network predictions, metrics, weekly features, figure
```

## Paper replication: OIES Energy Insight 177

[`paper_replication/`](paper_replication/) replicates *Momentum Trading and Managed Money Positioning in Energy: Relationships and Practical Applications* (Sun, Bouchouev and Fattouh, OIES Energy Insight 177, March 2026). It targets Insight 177 only, not every analysis in the earlier paper *Myths and Mysteries About Speculation in the Oil Market*.

### Scope and status

| Component | Status |
|---|---|
| Custom rolling futures contract, daily and cumulative P&L | Done (`data.py`) |
| Volatility-normalized momentum inputs, normalized Managed Money dependent variable | Implemented (`data.py`) |
| Simple weekly linear regression benchmark, rolling out-of-sample R² and directional accuracy | Done, matches the paper (see results) |
| Shared-signal, market-specific neural network | Implemented, results not yet valid (see below) |
| Novelty CTA vs discretionary position decomposition | Not started |

### Markets

| Key | Market | Venue | Contract size |
|---|---|---|---|
| `BRENT` | Brent crude oil | ICE Futures Europe | 1,000 barrels |
| `WTI` | WTI crude oil | CME/NYMEX | 1,000 barrels |
| `GASOIL` | Low-sulphur gasoil | ICE Futures Europe | 100 tonnes |
| `HEATOIL` | NY Harbor ULSD/heating oil | CME/NYMEX | 42,000 gallons |
| `RBOB` | RBOB gasoline | CME/NYMEX | 42,000 gallons |
| `NATGAS` | Henry Hub natural gas | CME/NYMEX | 10,000 MMBtu |

### Data

| File | Contents | Source | Coverage |
|---|---|---|---|
| `futures_generic_prices.csv` | Daily generic futures curve (`CL1`, `CO1`, ...; nearby 1-36): `PX_SETTLE`, `PX_LAST`, volume, open interest | Bloomberg | 2009-01-02 to 2026-09-01 |
| `contract_meta.csv` | Contract metadata including last tradeable and first notice dates | Bloomberg | Contracts expiring 2008-12 to 2029-09 |
| `managed_money.csv` | Weekly Managed Money long, short and net positions from the Commitments of Traders reports (CFTC for WTI, Heating Oil, RBOB, Natural Gas; ICE Futures Europe for Brent, Gasoil), futures-only and combined futures-and-options | Bloomberg | CFTC from 2009-01, ICE from 2011-01, to 2026-08-25 |
| `sg_cta_indices.csv` | SG CTA / Trend index levels and returns | SocGen API | 2000-01 to 2026-08 |

- **Managed Money basis:** futures-only by default; the combined futures-and-options panel is kept for sensitivity analysis.
- **Price field:** `PX_SETTLE` is the primary field. `PX_LAST` differs from it only on the final, not-yet-settled date (2026-09-01).
- **Nearby numbers:** `nearby` is the Bloomberg generic curve position (`CL1` = nearest unexpired contract). Generics roll to the next contract the day after the last tradeable date.

### Sample and evaluation windows

- **Paper comparison window:** sample from January 2011 to December 2025, with out-of-sample evaluation from January 2015 to December 2025. Each evaluation week is predicted by a model fitted only on the 104 weeks before it, so the first test week (2015-01-06) is trained on 2013-01-08 to 2014-12-30.
- **Expanded history:** every usable observation from 2009 to the latest available date. The earlier data warms up the 250-day momentum features and rolling volatility.

The two sets of outputs are kept separate. January-August 2026 has not been used by any results.

### Data processing

1. **Rolling contract** (`build_rolling_contract` in `data.py`). Following footnote 1 of the paper, each contract is rolled out of at the settlement of the last trading day of the month before its expiry month. Each market's own trading days are used, so exchange holidays are respected. Bloomberg generic nearby prices are mapped to specific contracts with the expiry schedule, so daily P&L and log returns are always measured on the contract actually held and never across a roll or expiry. The contract held at each close always expires in the calendar month after the next trading day. On the expiry day of contracts that expire on the last business day of a month (Heating Oil, RBOB, Brent since 2016), the position is briefly in nearby 3.
2. **Weekly panel.** Weeks are identified by `report_week_tuesday`. Returns are sampled at each COT as-of date (`report_date`: Tuesday, or Monday when Tuesday is a holiday). The dependent variable is the week-on-week change in futures-only Managed Money net positions. A week that straddles a year end is assigned to the year of the earliest as-of date across markets (e.g. 31 Dec 2018).
3. **Rolling out-of-sample regressions.** For each market and each week from 2015 to 2025, the change in positions is regressed on the weekly log return over the previous 104 weeks, with no intercept. The fitted slope is applied to the current week's return to predict that week's change. This is a nowcast: Tuesday's prices predict Tuesday's positions before the Friday COT release, as in the paper.
4. **Metrics.** Out-of-sample R² exactly as defined in the paper (1 − SSE / Σ dMM², measured against a prediction of zero change) and directional accuracy, by year and market and pooled across markets.

The paper does not state the intercept, return type, positions basis, exact roll timing or holiday-week dating. These were chosen as the options that reproduce the published tables. Because they were selected on the 2015-2025 evaluation period, the results below confirm the replication; they are not an independent test of the method. January-August 2026 could serve as a holdout.

### Results: simple linear regression benchmark vs the paper

Out-of-sample 2015-2025, cumulative by market (paper Tables 1 and 2, *Simple Linear Regression* panels):

| Market | R² replicated | R² paper | Accuracy replicated | Accuracy paper |
|---|---|---|---|---|
| ICE Brent | 27.14% | 27.10% | 69.69% | 69.34% |
| CME WTI | 11.52% | 11.52% | 72.30% | 72.30% |
| ICE Gasoil | 22.59% | 22.59% | 71.43% | 71.25% |
| CME Heating Oil | 10.77% | 10.77% | 64.46% | 64.46% |
| CME RBOB Gasoline | 17.98% | 17.98% | 67.25% | 67.25% |
| CME Natural Gas | 24.17% | 24.17% | 71.08% | 71.08% |
| **ALL Markets** | **21.10%** | **21.08%** | **69.37%** | **69.28%** |

All markets pooled, by year:

| Year | R² replicated | R² paper | Accuracy replicated | Accuracy paper |
|---|---|---|---|---|
| 2015 | 13.68% | 13.68% | 66.67% | 66.35% |
| 2016 | 23.36% | 23.27% | 70.83% | 70.51% |
| 2017 | 32.57% | 32.53% | 69.55% | 69.55% |
| 2018 | 19.89% | 19.89% | 67.30% | 67.30% |
| 2019 | 33.03% | 33.03% | 70.83% | 70.83% |
| 2020 | -45.90% | -45.90% | 67.31% | 67.31% |
| 2021 | 24.18% | 24.18% | 69.55% | 69.55% |
| 2022 | 0.23% | 0.23% | 66.03% | 66.03% |
| 2023 | 26.75% | 26.75% | 69.87% | 69.87% |
| 2024 | 41.85% | 41.85% | 74.84% | 74.53% |
| 2025 | 34.70% | 34.70% | 70.19% | 70.19% |

Across all 84 year-by-market cells:
- **R²:** the average absolute gap to the paper is 0.01 percentage points.
- **Directional accuracy:** the average gap is 0.09 pp. Every cell matches except Brent 2015, Brent 2016 and Gasoil 2024, each off by one week (±1.9 pp).
- **Where the R² gaps are:** all in Brent 2016-2017 (+0.27 and +0.18 pp), around January 2016, when two Brent contracts expired in the same month.

Full tables are in `outputs/linear_regression/r2_oos_vs_paper.csv` and `directional_accuracy_vs_paper.csv`.

### Neural network model (in progress)

`notebooks/replicate_insight_177.ipynb` runs a PyTorch implementation of the paper's main model on the same rolling contract, with two runs: the paper comparison window and the expanded history. Defaults are in `ReplicationConfig` (`config.py`):

- **Momentum inputs:** horizons of 5, 10, ..., 250 trading days. MOM = (cumulative P&L − its n-day moving average) / price volatility, then divided by its one-year rolling standard deviation.
- **Price volatility:** exponentially weighted with decay δ = 60/61 (Moskowitz, Ooi and Pedersen, 2011), annualized.
- **Dependent variable:** Managed Money minus its one-year rolling mean, divided by the rolling standard deviation of deviations from that mean, both lagged one week.
- **Shared layer:** 3 trend factors with reaction function R(u) = u·exp((1 − u²)/2), shared across markets.
- **Market layer:** market-specific weights plus a time-varying bias.
- **Loss:** mean squared error + L1 penalty λ₁ = 0.04 on the shared weights + L2 penalty λ₂ = 0.01 on weekly changes in the bias.
- **Training:** full-batch Adam with learning rate 0.01 on rolling 104-week windows. The first fit runs 1,024 epochs from shared weights of zero, market weights (0.4, 0.2, 0.1) and zero bias; later fits warm-start from the previous window for 36 epochs.

These results are not validated yet. The current saved outputs in `outputs/neural_network/` contain no valid predictions: predicted values are NaN, and the evaluation stops in November 2017.

### Setup and running

- **Requirements:** Python 3.11 or newer (developed on 3.14) with numpy, pandas, pyarrow, scipy, matplotlib and PyTorch (see `pyproject.toml`), plus Jupyter. TensorFlow and Gurobi are intentionally not required; PyTorch is the modeling library.
- **Running:** open the notebooks from `paper_replication/notebooks/`. Each notebook adds `src/` to the import path, so no install is needed. `linear_regression.ipynb` runs in seconds; `replicate_insight_177.ipynb` refits the network every week and takes much longer.
- **Torch import:** importing `paper_replication` imports torch through `__init__.py`, even for the linear regression notebook.

### Reproducibility rules

- Never difference a generic price series across a roll or expiry; compute P&L on the held contract and then cumulate it.
- Use `PX_SETTLE` as the primary price field.
- Use one Managed Money basis per run and record it in the configuration.
- Fit each evaluation date only on observations strictly before that date.
- Keep expanded-history and paper-window outputs separate.
- Record package versions, configuration, data file hashes and run timestamps with each result. This is not yet automated.
