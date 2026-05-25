"""
BAB (Betting Against Beta) Portfolio Analysis

Note this script will not work without the relevant data files (which I do not upload to GitHub)
Hence the purpose of this script is for illustrative purposes and to save the code in a public repo in case of local computer and backup issues.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.stats.sandwich_covariance import cov_hac
import statsmodels.api as sm

###############
# Set paths
##############
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# ── Constants ─────────────────────────────────────────────────────────────────
COUNTRY_CODES = [
    "GBR", "NLD", "DEU", "PRT", "USA", "NOR", "FIN", "FRA",
    "DNK", "CHE", "ITA", "AUT", "ESP", "IRL", "BEL", "SWE",
]
COUNTRY_NAMES = {
    "GBR": "United Kingdom", "NLD": "Netherlands", "DEU": "Germany",
    "PRT": "Portugal",       "USA": "United States","NOR": "Norway",
    "FIN": "Finland",        "FRA": "France",       "DNK": "Denmark",
    "CHE": "Switzerland",    "ITA": "Italy",        "AUT": "Austria",
    "ESP": "Spain",          "IRL": "Ireland",      "BEL": "Belgium",
    "SWE": "Sweden",
}

START_DATE = "1990-01"
END_DATE   = "2023-12"

########### 
# Helpers 
###########
def sharpe(x: pd.Series) -> float:
    """Annualised Sharpe ratio (monthly data → multiply by sqrt(12))."""
    return np.sqrt(12) * x.mean() / x.std()


def winsorize(series: pd.Series, lower: float = 0.001, upper: float = 0.999) -> pd.Series:
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)


def newey_west_ols(y: pd.Series, X: pd.DataFrame, nlags: int = 6):
    """
    Fit OLS and return result with Newey-West HAC standard errors, use 6 lags by default.
    """
    model = OLS(y, X, missing="drop").fit()
    nw_cov = cov_hac(model, nlags=nlags)
    return model.get_robustcov_results(cov_type="HAC", maxlags=nlags, use_correction=True)


def ols_alpha_beta(y: pd.Series, mkt: pd.Series) -> tuple[float, float]:
    """Return (alpha, beta) from a single-factor OLS."""
    X = add_constant(mkt, prepend=True)
    res = OLS(y, X, missing="drop").fit()
    return float(res.params.iloc[0]), float(res.params.iloc[1])


#################
# 1. Load & filter data 
##################

data = pd.read_csv("Data_complete_Bryan_Kelly_WRDS.csv")

# Drop indicator columns (already filtered on WRDS)
data = data.drop(columns=["obs_main", "exch_main", "primary_sec", "common"], errors="ignore")

# Keep selected countries
data = data[data["excntry"].isin(COUNTRY_CODES)].copy()

# Keep only relevant columns
COLS = [
    "source_crsp", "id", "permno", "gvkey", "eom", "excntry",
    "me", "ret", "ret_exc", "ret_exc_lead1m", "prc", "comp_exchg",
    "beta_60m", "beta_252d", "betabab_1260d", "fx", "curcd",
    "shares", "crsp_shrcd",
]
data = data[[c for c in COLS if c in data.columns]]

#################
#  2. Clean 
#################

# Keep only rows with price data
data = data.dropna(subset=["prc"])

# For CRSP (USA): keep share codes 10 & 11 only (set NaN → 0 first so non-CRSP rows pass)
data["crsp_shrcd"] = data["crsp_shrcd"].fillna(0)
data = data[data["crsp_shrcd"] != 12]

# Parse date
data["date"] = pd.to_datetime(data["eom"], format="%Y-%m-%d").dt.to_period("M")

# Lag beta_60m by 1 month within each stock
data = data.sort_values(["id", "date"])
data["l_beta_60m"] = data.groupby("id")["beta_60m"].shift(1)

################
#  3. Market-cap weights 
################

data["l_me"] = data.groupby("id")["me"].shift(1)
data = data.dropna(subset=["l_me"])   # Row needs to have lagged market cap.

data["weight"] = data.groupby(["date", "excntry"])["l_me"].transform(
    lambda x: x / x.sum()
)

################
#  4. Winsorize to reduce outliers
################

# Excess returns — monthly cross-section (to avoid forward-looking bias)
data["ret_exc"] = data.groupby("date")["ret_exc"].transform(
    lambda x: winsorize(x)
)

# Lagged beta — monthly cross-section
data["l_beta_60m"] = data.groupby("date")["l_beta_60m"].transform(
    lambda x: winsorize(x)
)

################
# 5. Market returns 
################

vw_mkt = (
    data.groupby(["date", "excntry"])
    .apply(lambda g: (g["weight"] * g["ret_exc"]).sum(), include_groups=False)
    .reset_index(name="vw_mkt_ret_exc")
)

n_stocks = data.groupby(["date", "excntry"])["id"].transform("nunique")
data["weight_eq"] = 1.0 / n_stocks

ew_mkt = (
    data.groupby(["date", "excntry"])
    .apply(lambda g: (g["weight_eq"] * g["ret_exc"]).sum(), include_groups=False)
    .reset_index(name="eq_mkt_ret_exc")
)

################
#  6. Summary-statistics table 
################

data_ss = data[
    (data["date"] >= pd.Period(START_DATE, "M")) &
    (data["date"] <= pd.Period(END_DATE, "M"))
].copy()

avg_me_by_date = (
    data_ss.groupby(["excntry", "date"])["me"]
    .mean()
    .reset_index()
    .groupby("excntry")["me"]
    .mean()
    .div(1_000)
    .rename("avg_me_company")
)

total_avg_me = (
    data_ss.groupby(["excntry", "date"])["me"]
    .sum()
    .reset_index()
    .groupby("excntry")["me"]
    .mean()
    .div(1_000)
    .rename("total_avg_me")
)

total_stocks = (
    data_ss.groupby("excntry")["id"]
    .nunique()
    .rename("total_n_stocks")
)

mean_stocks = (
    data_ss.groupby(["excntry", "date"])["id"]
    .count()
    .reset_index()
    .groupby("excntry")["id"]
    .mean()
    .round()
    .astype(int)
    .rename("mean_stocks")
)

date_range = data_ss.groupby("excntry")["date"].agg(["min", "max"]).rename(
    columns={"min": "start_date", "max": "end_date"}
)

summary_table = pd.concat(
    [total_stocks, mean_stocks, total_avg_me.round(), avg_me_by_date.round(2), date_range],
    axis=1,
).reset_index()

print("\n── Summary statistics ──────────────────────────────────────────────────")
print(summary_table.to_string(index=False))
summary_table.to_html("Descriptive_Statistics.html", index=False)

################
#  7. Create ercile portfolios 
################

# Require lagged beta (also drops Jan 1990)
data = data.dropna(subset=["l_beta_60m"])

# Shrink beta toward cross-sectional median (using methods from Frazzini-Pedersen)
data["l_beta_60m_shrunk"] = 0.6 * data["l_beta_60m"] + 0.4 * 1.0

# Assign tercile portfolio per country-month
data["portfolio"] = (
    data.groupby(["excntry", "date"])["l_beta_60m"]
    .transform(lambda x: pd.qcut(x, q=3, labels=[1, 2, 3]))
    .astype(int)
)

# Portfolio weights and returns
data["port_weight"] = data.groupby(["excntry", "date", "portfolio"])["l_me"].transform(
    lambda x: x / x.sum()
)
data["port_ret_exc"] = data.groupby(["excntry", "date", "portfolio"])[
    ["port_weight", "ret_exc"]
].transform(lambda g: g["port_weight"] * g["ret_exc"]).sum(axis=1)

# Calculate per group 
port_ret = (
    data.groupby(["excntry", "date", "portfolio"])
    .apply(
        lambda g: pd.Series({
            "port_ret_exc":      (g["port_weight"] * g["ret_exc"]).sum(),
            "portfolio_beta":    (g["port_weight"] * g["l_beta_60m"]).sum(),
            "portfolio_beta_shrunk": (g["port_weight"] * g["l_beta_60m_shrunk"]).sum(),
        }),
        include_groups=False,
    )
    .reset_index()
)

################
#  8. BAB (betting against beta) portfolio 
################

def make_bab(port_ret_df: pd.DataFrame) -> pd.DataFrame:
    """Build BAB from long (P1) minus short (P3) portfolio returns."""
    p1 = port_ret_df[port_ret_df["portfolio"] == 1].rename(
        columns={"port_ret_exc": "long",
                 "portfolio_beta": "beta1",
                 "portfolio_beta_shrunk": "beta1_shrunk"}
    ).drop(columns="portfolio")

    p2 = port_ret_df[port_ret_df["portfolio"] == 2].rename(
        columns={"port_ret_exc": "middle"}
    )[["excntry", "date", "middle"]]

    p3 = port_ret_df[port_ret_df["portfolio"] == 3].rename(
        columns={"port_ret_exc": "short",
                 "portfolio_beta": "beta3",
                 "portfolio_beta_shrunk": "beta3_shrunk"}
    ).drop(columns="portfolio")

    bab = p1.merge(p2, on=["excntry", "date"]).merge(p3, on=["excntry", "date"])
    bab["BAB_ret"] = bab["long"] - bab["short"]
    bab["BAB_ret_neutral"] = (
        (1 / bab["beta1_shrunk"]) * bab["long"]
        - (1 / bab["beta3_shrunk"]) * bab["short"]
    )
    return bab


bab = make_bab(port_ret)

# Merge with market returns
bab = bab.merge(vw_mkt, on=["excntry", "date"])
bab = bab.merge(ew_mkt,  on=["excntry", "date"])

# Apply date filter and require ≥15 stocks per country-month
stock_counts = data.groupby(["excntry", "date"])["id"].count().reset_index(name="total_stocks")
bab = bab.merge(stock_counts, on=["excntry", "date"])
bab = bab[bab["total_stocks"] >= 15]
bab = bab[
    (bab["date"] >= pd.Period(START_DATE, "M")) &
    (bab["date"] <= pd.Period(END_DATE, "M"))
].copy()

# Scale to percentages
PCT_COLS = ["long", "middle", "short", "BAB_ret", "BAB_ret_neutral",
            "vw_mkt_ret_exc", "eq_mkt_ret_exc"]
bab[PCT_COLS] = bab[PCT_COLS] * 100

################
# Compute 9. Cumulative returns 
################

for col in ["BAB_ret", "BAB_ret_neutral"]:
    bab[f"cum_{col}"] = bab.groupby("excntry")[col].transform(
        lambda x: (1 + x / 100).cumprod() - 1
    )

################
#  10. Aggregate regressions (Newey-West) 
################

def nw_summary(y_col: str, x_col: str, df: pd.DataFrame, label: str = ""):
    """Print Newey-West regression summary."""
    clean = df[[y_col, x_col]].dropna()
    X = add_constant(clean[x_col])
    res = OLS(clean[y_col], X).fit().get_robustcov_results(
        cov_type="HAC", maxlags=6, use_correction=True
    )
    print(f"\n── {label or y_col} ~ {x_col} ───────────────────────────────────")
    print(res.summary2().tables[1].to_string())
    return res


print("\n\n══ Aggregate results (VW market) ══════════════════════════════════════")
for col, lbl in [("long", "P1 (low beta)"), ("middle", "P2"),
                 ("short", "P3 (high beta)"), ("BAB_ret_neutral", "BAB neutral")]:
    nw_summary(col, "vw_mkt_ret_exc", bab, lbl)

# Average returns (intercept-only Newey-West)
print("\n── Average excess returns (Newey-West t-stats) ────────────────────────")
for col in ["long", "middle", "short", "BAB_ret_neutral", "BAB_ret"]:
    clean = bab[col].dropna()
    X = add_constant(pd.Series(np.ones(len(clean)), index=clean.index), has_constant="add")
    res = OLS(clean, X).fit().get_robustcov_results(
        cov_type="HAC", maxlags=6, use_correction=True
    )
    alpha, tval = res.params.iloc[0], res.tvalues.iloc[0]
    print(f"  {col:35s}  mean={alpha:6.3f}%  t={tval:5.2f}")

################
#  11. Sharpe ratios 
################
print("\n── Global Sharpe ratios ")
for col in ["long", "middle", "short", "BAB_ret", "BAB_ret_neutral"]:
    print(f"  {col:35s}  Sharpe={sharpe(bab[col].dropna()):.3f}")

################
# 12. Per-country results 
################

print("\n\n  Per-country results ")
country_results = []

for code in COUNTRY_CODES:
    sub = bab[bab["excntry"] == code].copy()
    if sub.empty:
        continue

    alpha, beta = ols_alpha_beta(sub["BAB_ret_neutral"], sub["vw_mkt_ret_exc"])
    sr = sharpe(sub["BAB_ret_neutral"])
    mean_ret = sub["BAB_ret_neutral"].mean()

    country_results.append({
        "excntry":    COUNTRY_NAMES[code],
        "excess_ret": round(mean_ret, 2),
        "Alpha":      round(alpha,    2),
        "Beta":       round(beta,     2),
        "Sharpe":     round(sr,       2),
    })

    # Newey-West per country (brief)
    X = add_constant(sub["vw_mkt_ret_exc"])
    res = OLS(sub["BAB_ret_neutral"], X, missing="drop").fit().get_robustcov_results(
        cov_type="HAC", maxlags=6, use_correction=True
    )
    print(f"\n  {COUNTRY_NAMES[code]} — alpha={alpha:.3f}%  beta={beta:.3f}  Sharpe={sr:.2f}")

results_df = pd.DataFrame(country_results).sort_values("excntry").reset_index(drop=True)
results_df.to_html("Table5.html", index=False)
print("\n── Per-country summary table ───────────────────────────────────────────")
print(results_df.to_string(index=False))

################
#  13. Plots 
################

def period_to_datetime(period_series: pd.Series) -> pd.Series:
    return period_series.dt.to_timestamp()


# Panel A – Sharpe ratios per country
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

ax1.bar(results_df["excntry"], results_df["Sharpe"], color="darkgreen", alpha=0.95, edgecolor="black")
ax1.set_ylim(0, 0.7)
ax1.set_xlabel("Country", fontweight="bold")
ax1.set_ylabel("Sharpe Ratio", fontweight="bold")
ax1.set_title("Panel A", loc="center")
ax1.tick_params(axis="x", rotation=45)

# Panel B – Alpha per country
ax2.bar(results_df["excntry"], results_df["Alpha"], color="#C04000", alpha=0.95, edgecolor="black")
ax2.set_ylim(0, 1.2)
ax2.set_xlabel("Country", fontweight="bold")
ax2.set_ylabel("Alpha", fontweight="bold")
ax2.set_title("Panel B", loc="center")
ax2.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("Figure_Sharpe_Alpha.png", dpi=150)
plt.close()

# Cumulative BAB return by country
fig, ax = plt.subplots(figsize=(12, 6))
for code in COUNTRY_CODES:
    sub = bab[bab["excntry"] == code].sort_values("date")
    if sub.empty:
        continue
    ax.plot(
        period_to_datetime(sub["date"]),
        sub["cum_BAB_ret"],
        label=code,
        linewidth=0.8,
    )
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Cumulative BAB Return by Country")
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative Return")
ax.legend(ncol=4, fontsize=7)
plt.tight_layout()
plt.savefig("Figure_Cumulative_BAB.png", dpi=150)
plt.close()

# USA detailed: cumulative returns for all portfolios
usa = bab[bab["excntry"] == "USA"].sort_values("date").copy()
for col in ["long", "middle", "short", "BAB_ret", "BAB_ret_neutral"]:
    usa[f"cum_{col}"] = (1 + usa[col] / 100).cumprod() - 1

fig, ax = plt.subplots(figsize=(12, 5))
palette = {"long": "red", "middle": "green", "short": "blue",
           "BAB_ret": "pink", "BAB_ret_neutral": "orange"}
labels  = {"long": "Low beta", "middle": "Mid beta", "short": "High beta",
           "BAB_ret": "BAB", "BAB_ret_neutral": "Market Neutral BAB"}
for col, clr in palette.items():
    ax.plot(period_to_datetime(usa["date"]), usa[f"cum_{col}"],
            color=clr, label=labels[col], linewidth=1)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("USA: Cumulative Returns by Portfolio")
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative Return")
ax.legend()
plt.tight_layout()
plt.savefig("Figure_USA_Portfolios.png", dpi=150)
plt.close()

print("\nDone. Outputs: Descriptive_Statistics.html, Table5.html, Figure_*.png")
