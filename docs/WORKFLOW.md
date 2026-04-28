# Example Financial Workflow: Fed → Market Transmission

This walkthrough builds a causal hypothesis about how Federal Reserve monetary
policy transmits through credit markets into equity volatility and returns.

The hypothesis: Fed Funds Rate → 10Y Treasury → Credit Spread → VIX → S&P 500

---

## Prerequisites

```bash
pip install -e ".[financial]"
export ANTHROPIC_API_KEY=sk-ant-...
export FRED_API_KEY=your-fred-key   # free at fred.stlouisfed.org/docs/api/api_key.html
```

Verify everything is wired up:

```bash
causal-engine doctor --financial
```

Expected output: all checks green except Python version if running 3.9 (non-blocking).

---

## Step 1: Build the causal DAG

Start a new session:

```bash
causal-engine new --name "Fed Market Transmission" --domain "macro finance" --adapter Financial
```

The DAGAgent opens an interactive session. A conversation might look like:

```
> What is the outcome variable you want to explain?

  S&P 500 weekly returns — I want to understand what drives them
  from the monetary policy side.

> Good. Let's start from the upstream cause. What is the root exposure?

  Fed Funds Rate — the overnight rate the Fed sets directly.

> Makes sense. Fed Funds Rate → what next?

  It affects the 10Y Treasury yield through expectations. Higher fed funds
  pushes up the long end of the curve.

> So: Fed Funds Rate → 10Y Treasury. What does 10Y Treasury cause?

  Credit spreads widen when yields rise — higher benchmark rate means
  riskier corporate debt has to offer more. So 10Y Treasury → Credit Spread.

> And Credit Spread → ?

  VIX. When credit stress rises, equity volatility follows. Then VIX → S&P 500.

/confirm
```

The agent summarises the DAG:

```
DAG: Fed Market Transmission
Nodes (5):
  Fed Funds Rate     [Exposure,   Hypothetical]
  10Y Treasury       [Mediator,   Hypothetical]
  Credit Spread      [Mediator,   Hypothetical]
  VIX                [Mediator,   Hypothetical]
  S&P 500            [Outcome,    Hypothetical]

Edges (4):
  Fed Funds Rate → 10Y Treasury
  10Y Treasury → Credit Spread
  Credit Spread → VIX
  VIX → S&P 500

Save this as version v1? [y/n] y
```

Note the version ID printed — e.g. `ver-8a3f21b9`. You will use it in the next steps.

---

## Step 2: Write the data manifest

Create `fed_market.yaml` in your working directory:

```yaml
adapter: financial
frequency: weekly
resample_method: last       # "last" is right for levels (yields, VIX, price index)
start_date: "2014-01-01"
end_date: "2024-12-31"

nodes:
  - label: "Fed Funds Rate"     # must match DAGVersion node label exactly
    source: fred
    series_id: "DFF"            # daily effective federal funds rate
    transform: diff             # first-difference to make stationary

  - label: "10Y Treasury"
    source: fred
    series_id: "DGS10"          # 10-year treasury constant maturity rate
    transform: diff

  - label: "Credit Spread"
    source: fred
    series_id: "BAMLH0A0HYM2"   # ICE BofA US high yield option-adjusted spread
    transform: diff

  - label: "VIX"
    source: yahoo
    ticker: "^VIX"
    transform: diff

  - label: "S&P 500"
    source: yahoo
    ticker: "^GSPC"
    transform: log_return       # log returns for price series
```

**Label matching is case-sensitive and exact.** `"S&P 500"` in the manifest must match
`"S&P 500"` in the DAGVersion node labels — not `"SP500"` or `"s&p 500"`.

**Choosing transforms:**

| Series type       | Transform     | Why                                         |
|-------------------|---------------|---------------------------------------------|
| Rate levels       | `diff`        | Rate changes are stationary; levels usually are not |
| Price index       | `log_return`  | Log returns are stationary and interpretable |
| Already stationary| `none`        | Skip if ADF confirms stationarity already   |

---

## Step 3: Fetch data and build the Parquet dataset

```bash
causal-engine dataset ver-8a3f21b9 --manifest fed_market.yaml
```

This runs the full pipeline:
1. Fetches each series from FRED and Yahoo Finance
2. Resamples to weekly frequency using the `last` observation
3. Applies declared transforms (`diff`, `log_return`)
4. Runs ADF stationarity tests on each column
5. If non-stationary after the declared transform, automatically applies one extra `diff` and re-tests
6. Writes a Parquet file to `~/.causal_engine/datasets/ver-8a3f21b9-fed_market-20240115T103042Z.parquet`

Expected output:

```
ADF Results
┌──────────────────┬──────────────────┬───────────────┬─────────┬────────────┐
│ Node             │ Transform Applied │ ADF Statistic │ p-value │ Stationary │
├──────────────────┼──────────────────┼───────────────┼─────────┼────────────┤
│ Fed Funds Rate   │ diff             │     -9.2341   │  0.0000 │     ✓      │
│ 10Y Treasury     │ diff             │     -8.7812   │  0.0000 │     ✓      │
│ Credit Spread    │ diff             │    -10.1234   │  0.0000 │     ✓      │
│ VIX              │ diff             │     -9.9871   │  0.0000 │     ✓      │
│ S&P 500          │ log_return       │    -12.3456   │  0.0000 │     ✓      │
└──────────────────┴──────────────────┴───────────────┴─────────┴────────────┘

✓ Dataset written to: /Users/you/.causal_engine/datasets/ver-8a3f21b9-fed_market-20240115T103042Z.parquet
  Columns (5): Fed Funds Rate, 10Y Treasury, Credit Spread, VIX, S&P 500
  Date range: 2014-01-05 → 2024-12-29  [weekly]
```

If a series fails the ADF test even after the extra diff, a `[WARN]` is printed but the
column is still included. Nothing is silently dropped.

---

## Step 4: Specify a custom output path

If you want the Parquet file in a known location for downstream analysis:

```bash
causal-engine dataset ver-8a3f21b9 --manifest fed_market.yaml --out ./data/features.parquet
```

---

## Step 5: Inspect the dataset

```python
import pandas as pd

df = pd.read_parquet("./data/features.parquet")
print(df.shape)          # (521, 5) — ~10 years of weekly observations
print(df.head())
print(df.corr())         # check expected correlations hold
```

The DataFrame index is a `DatetimeIndex` (weekly). Columns map one-to-one to manifest
node labels.

---

## Step 6: Modify the hypothesis with theory mode

After examining the data, you notice the Credit Spread → VIX link might be weaker
than expected. Start a modification session to refine the DAG:

```bash
causal-engine modify ver-8a3f21b9 --mode theory
```

The ModificationAgent reads the current DAG and proposes targeted changes.
Each proposed change is shown for your approval before anything is saved.
On confirmation, a new DAGVersion (e.g. `ver-9c4d30fa`) is created with
`parent_version_id = ver-8a3f21b9` and your modification rationale recorded.

---

## Step 7: Compare versions

```bash
causal-engine compare ver-8a3f21b9 ver-9c4d30fa
```

Output shows exactly which nodes and edges changed between the two versions,
colour-coded by add/remove/modify.

---

## Step 8: Build a new dataset for the modified version

Re-run the dataset command against the new version ID. The manifest file stays
the same unless you added or renamed nodes:

```bash
causal-engine dataset ver-9c4d30fa --manifest fed_market.yaml --out ./data/features_v2.parquet
```

---

## Common errors and fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `FRED API key missing` | `FRED_API_KEY` not set | `export FRED_API_KEY=...` |
| `Series "XYZ" not found on FRED` | Wrong `series_id` | Check at fred.stlouisfed.org |
| `No data returned for ticker "^VIX"` | yfinance rate-limit or bad ticker | Wait 60s, or check ticker at finance.yahoo.com |
| `Manifest node "X" not in DAGVersion` | Label mismatch | Make labels match exactly (case-sensitive) |
| `ADF test skipped (need >= 20)` | Too few observations after resample | Widen date range or use finer frequency |

To turn label mismatch warnings into hard errors (useful in CI or scripted pipelines):

```bash
causal-engine dataset ver-8a3f21b9 --manifest fed_market.yaml --strict
```

---

## Using a local CSV instead of FRED

If you have a downloaded data file or a proprietary signal, use `source: file`:

```yaml
nodes:
  - label: "Fed Funds Rate"
    source: file
    path: "./data/dff.csv"
    date_column: "DATE"
    value_column: "DFF"
    transform: diff
```

The CSV must have a parseable date column and a numeric value column.
Parquet files are also accepted (`.parquet` or `.pq` extension).

---

## Adding a direct-URL CSV source

For publicly available CSVs that can be fetched via HTTP GET:

```yaml
nodes:
  - label: "Inflation Expectations"
    source: url_csv
    url: "https://example.com/inflation_expectations.csv"
    date_column: "DATE"
    value_column: "VALUE"
    transform: none
```

HTML scraping (e.g. Wikipedia tables) is not supported in v1.
