# Capstone
## Repository structure

```text
Capstone/
├── README.md                                  # This file
├── working_ideas/                             # Extensions to the replicated model
│   ├── nn_carry_volatility.ipynb               # Carry and volatility experiments
│   ├── nn_benchmark.ipynb                      # SG returns as a benchmark for the NN's trend/bias split
│   ├── predict_sg_index.ipynb                  # Momentum layer trained on daily SG returns
│   ├── sg_input_feature.ipynb                  # SG-derived features as nowcast corrections
│   ├── Replicating_CTA.ipynb                   # Kestner (2020): SG Trend Index from a 16-futures momentum portfolio
│   ├── src/sg_cta.py                           # Helpers for the three SG notebooks
│   ├── src/kestner.py                          # Helpers for Replicating_CTA.ipynb
│   ├── tests/test_kestner.py                   # Roll, signal and timing tests for kestner.py
│   └── data_required/                          # Cross-asset CTA data (see below)
│       ├── futures_generic_prices.csv.gz       # Daily generic curves, 50 futures markets (Bloomberg)
│       ├── futures_contract_chain.csv          # Every contract with its last trade date: the roll schedule
│       ├── futures_markets.csv                 # One row per futures market: sector, root, cycle, currency, ...
│       ├── swap_rates.csv.gz                   # 10Y and 7Y swap rates, 14 currencies (Bloomberg)
│       ├── money_market_3m.csv.gz              # 3-month fixings, 14 currencies (Bloomberg)
│       ├── fx_spot.csv.gz                      # FX spot, 30 currencies against USD (Bloomberg)
│       ├── fx_forward_3m.csv.gz                # FX 3-month forwards (Bloomberg)
│       ├── lme_forwards.csv.gz                 # LME nickel and aluminium, cash to 27 months (Bloomberg)
│       ├── equity_indices.csv.gz               # 26 cash indices: level, dividend yield, total return (Bloomberg)
│       ├── macro_cpi.csv                       # CPI, 31 economies + Taiwan (IMF, Eurostat, Bloomberg)
│       └── macro_gdp.csv                       # Quarterly nominal GDP, 15 economies (IMF)
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

## Working ideas

In [`nn_carry_volatility.ipynb`](working_ideas/nn_carry_volatility.ipynb), we test direct carry, carry correction, volatility scaling, and scaling plus carry correction, with yearly results.

### Results (2017–2025)

We compare all models on the same 2,820 market-week observations. These are retrospective estimates of same-week position changes using current prices.

| Model | R² | R² gain vs NN (pp) | Directional accuracy | MAE (contracts) |
|---|---|---|---|---|
| Original NN | 44.90% | — | 72.70% | 10,794 |
| Direct carry | 46.93% | +2.03 | 73.48% | 10,635 |
| Carry correction | 46.02% | +1.12 | 73.40% | 10,707 |
| Volatility scaling | 47.27% | +2.37 | 73.30% | 10,600 |
| **Scaling + carry correction** | **48.40%** | **+3.49** | **73.55%** | **10,499** |

Three notebooks use SG Prime Services index returns, to be read in order:

- [`nn_benchmark.ipynb`](working_ideas/nn_benchmark.ipynb): SG Trend returns as a benchmark for the NN's split of Managed Money into trend (CTA) and bias (discretionary) positions. Energy P&L on trend positions correlates 0.35 with weekly SG Trend returns (rank 0.32); bias-position P&L only 0.06 by rank.
- [`predict_sg_index.ipynb`](working_ideas/predict_sg_index.ipynb): the paper's momentum layer trained on daily SG Trend returns. Out-of-sample daily correlation 0.33 (2015–2025), below the fixed 20/120 rule (0.36).
- [`sg_input_feature.ipynb`](working_ideas/sg_input_feature.ipynb): SG-implied flow and daily SG residual features, as weekly summaries or day by day, as corrections to the NN nowcast and as a one-week-ahead forecast. None beats the original NN (44.90% R², 2017–2025); weekly features change R² by −0.20 to −5.41 pp, day-by-day inputs by −2.32 to −21.65 pp.

### Replicating CTA returns across asset classes (Kestner, 2020)

[`Replicating_CTA.ipynb`](working_ideas/Replicating_CTA.ipynb) replicates *Replicating CTA Positioning: An Improved Method* (Kestner, 2020) on the cross-asset data below. A volatility-scaled momentum portfolio on 16 futures (equity indices, bonds, currencies, commodities) explains weekly SG Trend Index returns, and its positions estimate what trend followers hold.

| Result | Replication | Paper |
|---|---|---|
| R² of the ensemble (16-, 32- and 52-week lookbacks), 2015–2019 | 0.77 | above 0.75 |
| R² of the 16/1/90, 32/1/90 and 52/1/90 models, 2015–2019 | 0.61, 0.67, 0.62 | ≈ 0.61, 0.67, 0.62 |
| Mean gap to the paper's 15 sensitivity bars | 0.005 | — |
| R² of the ensemble out of sample, January 2020 to August 2026 | 0.64 (0.53–0.77 by year) | — |

- **Exposures** are on the paper's scale: up to 381% of capital in bonds, and between −$145bn and +$332bn in the four equity markets at $300bn of trend-following assets.
- **A 20-week rolling regression lags the replication:** its S&P 500 beta best matches the replication's equity exposure from 12 weeks earlier.
- **Choices the paper leaves open** (roll timing, how the ensemble averages its models) move the paper-window R² by at most 0.005. The notebook rolls five trading days before expiry.

Run it from `working_ideas/`; it takes about 10 seconds. Figures and tables are saved to `paper_replication/outputs/replicating_cta/`.

## Cross-asset CTA data (`working_ideas/data_required/`)

Data for explaining CTA returns with momentum (and carry and value) portfolios across asset classes, built for two papers:

- Kestner (2020), *Replicating CTA Positioning: An Improved Method*: 16 futures in equities, bonds, currencies and commodities.
- Baz, Granger, Harvey, Le Roux and Rattray (2015), *Dissecting Investment Strategies in the Cross Section and Time Series*: carry, momentum and value on equity index futures, commodities, currencies and interest rate swaps, 1990–2015.

The CTA returns to explain, the SG CTA and SG Trend indices, stay in `paper_replication/data_required/sg_cta_indices.csv`.

**Sources.** A Bloomberg Terminal pull on 2026-10-05, daily from 1988-01-01, for everything except CPI and GDP, which come from DBnomics (IMF International Financial Statistics; euro-area HICP from Eurostat). Bloomberg data are proprietary.

### Files

| File | Contents | Coverage |
|---|---|---|
| `futures_generic_prices.csv.gz` | Generic futures curves (`ES1`, `ES2`, ...) for 50 markets: 28 equity index, 4 bond, 4 currency, 14 commodity. `PX_SETTLE`, `PX_LAST`, volume, open interest; same columns as `paper_replication`'s file. Commodities go 3 years out (`CL1`–`CL36`), financial futures about a year | 1988 (or listing) to 2026-10-05 |
| `futures_contract_chain.csv` | Every contract of each market, expired ones included: delivery year and month, last trade date, and `in_generic_cycle` (whether the generic series steps through it) | 14,712 contracts |
| `futures_markets.csv` | One row per market: name, sector, exchange, Bloomberg root and yellow key, generic cycle, currency, value of one point, data range, matching cash index, paper(s) using it, notes | 50 markets |
| `swap_rates.csv.gz` | 10- and 7-year swap rates in percent (Bloomberg `CMPN` New York close), 14 currencies | From 1988–2001 by currency |
| `money_market_3m.csv.gz` | 3-month fixings in percent (LIBOR, Euribor, BBSW, CDOR, NIBOR, STIBOR, ...), 14 currencies | From 1988–2001 |
| `fx_spot.csv.gz` | Spot against USD, 30 currencies, with the quote convention | From 1988 (EM from 1991–1993) |
| `fx_forward_3m.csv.gz` | 3-month forward points (non-deliverable forwards for KRW, INR, IDR, MYR, PHP, TWD, BRL, CLP, COP, PEN), Bloomberg's scale, outright forward | From 1988 (EM from 1995–2005) |
| `lme_forwards.csv.gz` | LME nickel and aluminium: cash, 3, 15 and 27 months | From 1988 |
| `equity_indices.csv.gz` | 26 cash indices: level, 12-month dividend yield in percent, gross total-return index | Levels from 1988; dividend yields mostly from 2000–2002 |
| `macro_cpi.csv` | CPI for the 30 currencies' economies and the US (IMF; euro area from Eurostat, Germany kept as the pre-1996 proxy; Taiwan from Bloomberg) | Mostly 1950s to mid-2025 |
| `macro_gdp.csv` | Quarterly nominal GDP in local currency, the 14 swap currencies plus Germany (IMF) | From 1950–1995 to 2025 |

### How to use it

- **Prices:** use `PX_SETTLE`, falling back on `PX_LAST`.
- **Rolls:** Bloomberg's generics move to the next contract on the first trading day after the front contract's last trade date. Take the last trade dates from `futures_contract_chain.csv` with `in_generic_cycle` true. On a roll day, measure the return on the contract held, `G1(t) / G2(t-1) - 1`, never `G1(t) / G1(t-1) - 1`.
- **Carry:** equity carry uses `G1` and `G2` with their last trade dates. Commodity carry uses the contract expiring a year after the front, generic number `1 + len(generic_cycle)` (`CL13`, `GC7`).
- **FX:** `quote` is "USD per currency" for EUR, GBP, AUD and NZD and "currency per USD" otherwise. `forward = spot + forward_points / 10**fwd_scale`. Carry for a long position in the currency is `4 * (spot / forward - 1)` in USD per currency, `4 * (forward / spot - 1)` otherwise.
- **Macro:** dates are the first day of the reference period, not release dates. Baz et al. lag GDP three months.

### Checks run on this data

- **Kestner (2020):** an ensemble of the 16-, 32- and 52-week momentum models explains weekly SG Trend Index returns with R² 0.75 over 2007–2019 and 0.77 over 2015–2019. The paper reports above 75%. Over 2015–2019 the single models give 0.60, 0.67 and 0.62, against about 0.61, 0.67 and 0.62 in its charts.
- **Baz et al. (2015):** the first date each signal can be computed matches the paper's appendix for rates carry (all 14 currencies, to the day), FX carry, commodity carry and equity carry.
- **Underlyings:** every equity future tracks its cash index (daily return correlation 0.85–0.99), the currency futures track FX spot, and CL is identical to `paper_replication`'s energy data.
- **Rolls:** open interest confirms the roll the day after the last trade date in about 100% of testable cases for almost every market (OBX 79%).
- **FX forwards:** carry correlates 0.96–1.00 with the 3-month rate differential (PHP 0.80).

### Known issues

- **No Finland futures.** The pulled `HX1 Index` is a USD contract on another exchange, not the OMX Helsinki 25 future, and is left out. The `HEX25` cash index is in `equity_indices.csv.gz`.
- **Non-positive values.** Drop them, except WTI's genuine settlement of −37.63 on 2020-04-20. The others are zeros in far contracts (FTASE nearby 2–4 in 2010, silver nearby 10 and platinum nearby 6 in October 2016), the Bovespa level before 1989-12-21, and New Zealand CPI in 1920–1925.
- **Gaps.** No SMI futures from 1994-12-16 to 1997-12-22; OBX futures are sparse in 1993–1994; FTASE has gaps in spring 2009 and in July 2015, when the Athens exchange closed.
- **Platinum and palladium.** The contract a year out is mostly missing over 1992–2009. Compute their carry from the furthest available contract, annualised by the time between expiries.
- **Big US index contracts.** `SP`, `ND` and `MD` give the history before the E-minis. `MD` has only its front month, so MidCap carry starts with the E-mini in 2002. When `SP` was delisted on 2021-09-17, every remaining contract got that last trade date.
- **Dividend yields** start in 2000–2002 for most indices (2004 Nikkei, 2013 Hang Seng). They can be extended from the total-return and price indices, except for DAX, BUX, Bovespa and OBX, whose levels already include dividends.
- **Series that stop.** The LIBOR-based swaps and fixings end between 2021 and 2024, and the IMF CPI and GDP in 2025. This does not affect the papers' samples.
- **Not available.** Baz et al.'s history before 1988 (Man AHL and Global Financial Data), and the nine indices the paper lists but leaves out of its 26-index table.

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

### Neural network model

`notebooks/nn_replicate_insight_177.ipynb` runs a PyTorch implementation of the paper's main model on the same rolling contract, with two runs: the paper comparison window and the expanded history. Defaults are in `ReplicationConfig` (`config.py`):

- **Momentum inputs:** horizons of 5, 10, ..., 250 trading days. MOM = (cumulative P&L − its n-day moving average) / price volatility, then divided by its one-year rolling standard deviation.
- **Price volatility:** exponentially weighted with decay δ = 60/61 (Moskowitz, Ooi and Pedersen, 2011), annualized.
- **Dependent variable:** Managed Money minus its one-year rolling mean, divided by the square root of summed historical residual squares / (N−1), both lagged one report. Windows cover 252 trading dates; residuals are not re-centered. See [NN alignment notes](paper_replication/NN_PAPER_ALIGNMENT.md).
- **Shared layer:** 3 trend factors with reaction function R(u) = u·exp((1 − u²)/2), shared across markets.
- **Market layer:** market-specific weights plus a time-varying bias.
- **Loss:** mean squared error + L1 penalty λ₁ = 0.04 on the shared weights + L2 penalty λ₂ = 0.01 on weekly changes in the bias.
- **Training:** full-batch Adam with learning rate 0.01 on rolling 104-week windows. The first fit runs 1,024 epochs from shared weights of zero, market weights (0.4, 0.2, 0.1) and zero bias; later fits warm-start from the previous window for 36 epochs.

#### Replication results (2015–2025)

Cumulative results for 574 reporting weeks per market (3,444 observations). Paper values are from the **Our Model** panels of Tables 1–2. All-market metrics pool observations rather than average market scores.

| Market | R² replicated | R² paper | Accuracy replicated | Accuracy paper |
|---|---|---|---|---|
| ICE Brent | 49.17% | 48.34% | 73.52% | 74.04% |
| CME WTI | 42.03% | 41.86% | 74.39% | 75.96% |
| ICE Gasoil | 46.75% | 45.76% | 73.87% | 73.87% |
| CME Heating Oil | 36.38% | 33.74% | 67.94% | 68.12% |
| CME RBOB Gasoline | 27.71% | 33.99% | 68.82% | 69.69% |
| CME Natural Gas | 44.21% | 43.69% | 72.47% | 73.34% |
| **ALL Markets** | **45.03%** | **44.59%** | **71.84%** | **72.50%** |

All markets pooled, by year:

| Year | R² replicated | R² paper | Accuracy replicated | Accuracy paper |
|---|---|---|---|---|
| 2015 | 33.02% | 30.39% | 67.63% | 70.51% |
| 2016 | 46.97% | 51.44% | 69.87% | 72.76% |
| 2017 | 46.16% | 49.56% | 71.79% | 73.40% |
| 2018 | 45.02% | 40.02% | 73.90% | 75.47% |
| 2019 | 47.72% | 45.21% | 79.49% | 79.49% |
| 2020 | 27.03% | 26.72% | 66.35% | 66.99% |
| 2021 | 35.84% | 40.79% | 68.91% | 68.27% |
| 2022 | 37.94% | 28.10% | 69.87% | 68.27% |
| 2023 | 47.21% | 47.99% | 70.51% | 71.47% |
| 2024 | 57.09% | 54.37% | 78.62% | 78.30% |
| 2025 | 49.75% | 49.96% | 73.08% | 72.44% |

The pooled results are close, but this is not an exact numerical replication: cumulative RBOB R² is 6.28 pp lower, while annual differences include Gasoil R² at +30.69 pp in 2022 and Brent directional accuracy at −11.32 pp in 2018. Aggregation can conceal these differences across markets and years.

Possible contributors include data revisions, holiday alignment, window boundaries, and implementation choices the paper does not fully specify: bias initialization, optimizer-state resets, and which fitted model supplies the prior prediction endpoint. Small changes can propagate through successive nonlinear fits; we have not isolated each contributor, so these are hypotheses rather than established causes. Full annual comparisons and charts are at the end of `notebooks/nn_replicate_insight_177.ipynb`.

### Setup and running

- **Requirements:** Python 3.11 or newer (developed on 3.14) with numpy, pandas, pyarrow, scipy, matplotlib and PyTorch (see `pyproject.toml`), plus Jupyter. TensorFlow and Gurobi are intentionally not required; PyTorch is the modeling library.
- **Running:** open the notebooks from `paper_replication/notebooks/`. Each notebook adds `src/` to the import path, so no install is needed. `linear_regression.ipynb` runs in seconds; `nn_replicate_insight_177.ipynb` refits the network every week and takes much longer.
- **Torch import:** importing `paper_replication` imports torch through `__init__.py`, even for the linear regression notebook.

### Reproducibility rules

- Never difference a generic price series across a roll or expiry; compute P&L on the held contract and then cumulate it.
- Use `PX_SETTLE` as the primary price field.
- Use one Managed Money basis per run and record it in the configuration.
- Fit each evaluation date only on observations strictly before that date.
- Keep expanded-history and paper-window outputs separate.
- Record package versions, configuration, data file hashes and run timestamps with each result. This is not yet automated.
