# Insight 177 Paper Replication

A PyTorch implementation plan and data package for reproducing **OIES Energy Insight 177**, *Momentum Trading and Managed Money Positioning in Energy: Relationships and Practical Applications* (Sun, Bouchouev and Fattouh, March 2026).

The project uses the full six-market energy-futures universe available in the source data and preserves the paper's 2011-2025 evaluation period as a named comparison slice.

## Scope

This project targets Insight 177 only. It does not attempt to reproduce every analysis in the earlier paper *Myths and Mysteries About Speculation in the Oil Market*.

The replication will produce:

1. The paper's custom rolling futures-contract series.
2. Daily contract P&L and cumulative P&L.
3. Volatility-normalized momentum inputs.
4. The normalized Managed Money dependent variable.
5. The shared-signal, market-specific allocation model.
6. The simple weekly linear benchmark.
7. Rolling out-of-sample forecasts, R-squared, and directional accuracy.
8. CTA-versus-discretionary position decompositions.

## Data

The `data_required/` directory is copied from `owen/owen/data_required/` and contains:

| File | Role |
|---|---|
| `futures_generic_prices.csv` | Daily generic futures curve, including settlement prices and nearby numbers. |
| `contract_meta.csv` | Contract metadata and last-tradeable dates used to determine the custom roll. |
| `managed_money.csv` | Weekly Managed Money positioning for all six markets. |
| `sg_cta_indices.csv` | SG CTA/Trend index levels and returns; retained as an optional extension input. |

The source data currently cover prices through 2026-09-01, Managed Money observations through 2026-08-25, contract metadata through 2029-09-12, and SG index data through 2026-08-31.

The project uses the **futures-only** Managed Money panel by default. The combined futures-and-options panel remains available for sensitivity analysis.

## Markets

The model covers all six markets in Insight 177:

| Key | Market | Venue | Contract size |
|---|---|---|---|
| `WTI` | WTI crude oil | CME/NYMEX | 1,000 barrels |
| `BRENT` | Brent crude oil | ICE Futures Europe | 1,000 barrels |
| `GASOIL` | Low-sulphur gasoil | ICE Futures Europe | 100 tonnes |
| `HEATOIL` | NY Harbor ULSD/heating oil | CME/NYMEX | 42,000 gallons |
| `RBOB` | RBOB gasoline | CME/NYMEX | 42,000 gallons |
| `NATGAS` | Henry Hub natural gas | CME/NYMEX | 10,000 MMBtu |

The full available generic curve is retained. The paper's custom roll uses nearby 1 and nearby 2, selected using the contract expiry metadata; deeper curve contracts remain available for diagnostics and future extensions.

## History and evaluation

Two windows are recorded explicitly:

- **Expanded history:** use every usable observation in the supplied files, beginning in 2009 and extending through the latest available date. Earlier data provide warm-up for the 250-trading-day momentum features and rolling volatility estimates.
- **Paper comparison window:** January 2011 through December 2025, with out-of-sample evaluation from January 2015 through December 2025.

The expanded run is the primary requested run. The paper comparison window is required so results can be compared with Insight 177 without silently changing the published sample.

## Model parameters

Defaults are documented in `SPEC.md` and should be centralized in a configuration object when implementation begins.

- Momentum horizons: `5, 10, 15, ..., 250` trading days.
- Price-volatility decay: `delta = 60/61`, following Moskowitz, Ooi and Pedersen (2011).
- Momentum normalization: divide the volatility-scaled momentum by its one-year rolling standard deviation.
- Managed Money normalization: one-year rolling mean and standard deviation, lagged by five trading days/one reporting week.
- Hidden factors: `3` shared trend-following factors.
- Reaction function: `R(u) = u * exp((1 - u^2) / 2)`.
- Training window: two years of weekly Managed Money observations.
- First rolling fit: `1,024` full-batch epochs.
- Later rolling fits: `36` full-batch warm-start epochs.
- Optimizer: PyTorch Adam, learning rate `0.01`.
- L1 penalty on shared first-layer weights: `lambda_1 = 0.04`.
- L2 penalty on weekly changes in the time-varying bias: `lambda_2 = 0.01`.
- Initialization: deterministic; first fit uses shared weights zero, market weights `(0.4, 0.2, 0.1)`, and zero bias. Later fits warm-start from the previous window.
- Benchmark: per-market linear regression of weekly Managed Money changes on the corresponding weekly rolling-contract return.
- Metrics: pooled and market-level out-of-sample R-squared and directional accuracy.

## Software

The implementation uses:

- Python 3.11 or newer
- PyTorch for the neural-network model and automatic differentiation
- pandas and NumPy for data preparation
- SciPy for numerical utilities where needed
- scikit-learn only for simple benchmark utilities if useful; the benchmark should remain transparent and reproducible
- matplotlib for diagnostic figures

TensorFlow and Gurobi are intentionally not required. Gurobi is not a natural fit for the paper's Adam-trained nonlinear model, and PyTorch is the requested modeling library.

## Planned layout

```text
paper_replication/
├── README.md
├── SPEC.md
├── pyproject.toml
├── data_required/
├── src/
│   └── paper_replication/
├── tests/
└── outputs/
```

## Reproducibility rules

- Do not difference a generic series across a roll. Compute daily P&L within the selected contract and then cumulate P&L.
- Use `PX_SETTLE` as the primary price field.
- Join weekly Managed Money observations using `report_week_tuesday`, not only the raw published report date, because CFTC and ICE holidays can fall on different reporting dates.
- Select one Managed Money basis for a run and record it in the configuration and output metadata.
- Fit each evaluation date only on observations strictly before that date.
- Keep the expanded-history and exact-paper evaluation outputs separate.
- Record package versions, configuration, data file hashes, and run timestamps with every result.

## Current status

The raw data have been copied into this project. The next implementation slice is the rolling-contract and feature-construction pipeline, followed by the benchmark and then the PyTorch rolling model.
