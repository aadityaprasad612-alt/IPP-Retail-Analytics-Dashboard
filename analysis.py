"""analysis.py - data loading, cleaning and the regression methodology from the IPP notebook."""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

DATA_URL = ("https://raw.githubusercontent.com/EmpoweredAnalysts/FabricRetailSales/"
            "main/retail_sales_demo_data.csv")
LOCAL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retail_sales_demo_data.csv")

FACTORS1: List[str] = ["ad_spend", "promo_days", "competitors_within_3km",
                       "parking_spaces", "store_age_years", "is_holiday_month"]
FACTORS2: List[str] = ["avg_price", "staff_count", "inventory_fill_rate", "returns_rate",
                       "promo_days", "local_income_index", "is_holiday_month"]
FACTORS2_WITH_AD: List[str] = ["ad_spend"] + FACTORS2          # C3 check: Stage 2 + ad spend
SALES_LEVERS: List[str] = ["ad_spend", "promo_days", "avg_price", "staff_count",
                           "inventory_fill_rate", "is_holiday_month"]
FOOTFALL_LEVERS: List[str] = ["ad_spend", "promo_days", "is_holiday_month"]
CONTROLLABLE2: List[str] = ["ad_spend", "avg_price", "promo_days", "staff_count", "inventory_fill_rate"]
LEVER_STEP = {"ad_spend": 1000, "avg_price": 1, "promo_days": 1, "staff_count": 1, "inventory_fill_rate": 0.01}
LEVER_LABEL = {"ad_spend": "+1,000 ad spend", "avg_price": "+1 in average price",
               "promo_days": "+1 promotion day", "staff_count": "+1 staff member",
               "inventory_fill_rate": "+1 percentage point stock availability"}

LABELS = {
    "foot_traffic": "Footfall", "monthly_sales": "Monthly sales", "ad_spend": "Ad spend",
    "promo_days": "Promo days", "competitors_within_3km": "Competitors (3 km)",
    "parking_spaces": "Parking spaces", "store_age_years": "Store age (yrs)",
    "is_holiday_month": "Holiday month", "avg_price": "Avg price", "staff_count": "Staff count",
    "inventory_fill_rate": "Inventory fill rate", "returns_rate": "Returns rate",
    "local_income_index": "Local income index", "store_size_sqft": "Store size (sqft)",
}

REQUIRED_COLUMNS = ["store_id", "year", "month", "foot_traffic", "monthly_sales", "ad_spend",
                    "is_holiday_month", "store_size_sqft"] + sorted(set(FACTORS1 + FACTORS2))


def label(col: str) -> str:
    return LABELS.get(col, col)


# ----------------------------------------------------------------------------- loading
def load_raw(source: str | None = None) -> pd.DataFrame:
    """Load CSV from a given path/URL, else local file next to the app, else GitHub raw URL."""
    candidates = [source] if source else [LOCAL_FILE, DATA_URL]
    last_err: Exception | None = None
    for src in candidates:
        try:
            if src and (src.startswith("http") or os.path.exists(src)):
                return pd.read_csv(src)
        except Exception as err:  # keep trying the next source
            last_err = err
    raise RuntimeError(f"Could not load data from {candidates}: {last_err}")


def validate(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def clean_data(raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Notebook A3: fix impossible footfall (>100,000) with store median; fill ad_spend with store median."""
    validate(raw)
    df = raw.copy()
    report = {"rows": len(df), "stores": df["store_id"].nunique(),
              "duplicates": int(df.duplicated().sum()),
              "missing_ad_spend": int(df["ad_spend"].isna().sum())}
    bad = df["foot_traffic"] > 100000
    report["impossible_footfall_fixed"] = int(bad.sum())
    store_median = df[~bad].groupby("store_id")["foot_traffic"].median()
    df.loc[bad, "foot_traffic"] = df.loc[bad, "store_id"].map(store_median)
    df["ad_spend"] = df["ad_spend"].fillna(df.groupby("store_id")["ad_spend"].transform("median"))
    report["missing_after"] = int(df.isna().sum().sum())
    return df, report


# ----------------------------------------------------------------------------- models
def standardised_regression(data: pd.DataFrame, cols: List[str], target: str) -> Tuple[pd.Series, float]:
    """Z-score everything, least squares with NumPy, return coefficients and R-squared."""
    X, y = data[cols], data[target]
    X_z = (X - X.mean()) / X.std()
    y_z = (y - y.mean()) / y.std()
    A = np.column_stack([np.ones(len(X_z)), X_z])
    coefs = np.linalg.lstsq(A, y_z, rcond=None)[0]
    pred = A @ coefs
    r2 = 1 - ((y_z - pred) ** 2).sum() / ((y_z - y_z.mean()) ** 2).sum()
    return pd.Series(coefs[1:], index=cols), float(r2)


def raw_fit(data: pd.DataFrame, cols: List[str], target: str) -> Tuple[np.ndarray, np.ndarray]:
    A = np.column_stack([np.ones(len(data)), data[cols]])
    coefs = np.linalg.lstsq(A, data[target], rcond=None)[0]
    return coefs, A @ coefs


def within_effects(data: pd.DataFrame, target: str, levers: List[str]) -> Tuple[pd.Series, float]:
    """Effect of each lever INSIDE a store (store means removed), in real units, plus R-squared."""
    d = data.copy()
    for col in levers + [target]:
        d[col] = d[col] - d.groupby("store_id")[col].transform("mean")
    A = np.column_stack([np.ones(len(d)), d[levers]])
    coefs = np.linalg.lstsq(A, d[target], rcond=None)[0]
    r2 = 1 - ((d[target] - A @ coefs) ** 2).sum() / ((d[target] - d[target].mean()) ** 2).sum()
    return pd.Series(coefs[1:], index=levers), float(r2)


def ranking_table(beta: pd.Series) -> pd.DataFrame:
    rank = beta.abs().sort_values(ascending=False)
    pct = (rank / rank.sum() * 100).round(1)
    out = pd.DataFrame({"coefficient": beta[rank.index].round(3), "importance_%": pct})
    out.index.name = "factor"
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


def correlations(df: pd.DataFrame, factors: List[str], target: str) -> pd.Series:
    return df[factors + [target]].corr()[target].drop(target).sort_values(ascending=False)


def footfall_sales_link(df: pd.DataFrame) -> pd.Series:
    """Notebook C1a."""
    avg = df.groupby("store_id")[["foot_traffic", "monthly_sales"]].mean()
    nh, h = df[df["is_holiday_month"] == 0], df[df["is_holiday_month"] == 1]
    return pd.Series({
        f"All rows ({len(df)})": df["foot_traffic"].corr(df["monthly_sales"]),
        "Non-holiday months only": nh["foot_traffic"].corr(nh["monthly_sales"]),
        "Holiday months only": h["foot_traffic"].corr(h["monthly_sales"]),
        f"Store averages ({len(avg)} stores)": avg["foot_traffic"].corr(avg["monthly_sales"]),
    }).round(2)


def lever_table(df: pd.DataFrame, sales_effects: pd.Series) -> pd.DataFrame:
    """Notebook C4."""
    swing = {c: (df[c] - df.groupby("store_id")[c].transform("mean")).std() for c in CONTROLLABLE2}
    out = pd.DataFrame({
        "change": [LEVER_LABEL[c] for c in CONTROLLABLE2],
        "extra_sales_per_month": [round(sales_effects[c] * LEVER_STEP[c], 1) for c in CONTROLLABLE2],
        "typical_swing_in_a_store": [round(swing[c], 2) for c in CONTROLLABLE2],
    }, index=CONTROLLABLE2).sort_values("extra_sales_per_month", ascending=False)
    out.index.name = "lever"
    return out


def run_models(df: pd.DataFrame) -> Dict[str, object]:
    """All model results used by the dashboard (fitted on the full cleaned data, as in the notebook)."""
    out: Dict[str, object] = {}
    out["beta1"], out["r2_1"] = standardised_regression(df, FACTORS1, "foot_traffic")
    out["result1"] = ranking_table(out["beta1"])
    coefs1, _ = raw_fit(df, FACTORS1, "foot_traffic")
    out["effect1"] = pd.Series(coefs1[1:], index=FACTORS1)

    out["beta2"], out["r2_2"] = standardised_regression(df, FACTORS2, "monthly_sales")
    out["result2"] = ranking_table(out["beta2"])
    coefs2, _ = raw_fit(df, FACTORS2, "monthly_sales")
    out["effect2"] = pd.Series(coefs2[1:], index=FACTORS2)

    # C3 check: add ad spend to the sales model (32% -> 56%)
    out["beta_ad"], out["r2_ad"] = standardised_regression(df, FACTORS2_WITH_AD, "monthly_sales")
    out["check_ad"] = ranking_table(out["beta_ad"])
    all_factors = list(dict.fromkeys(FACTORS2 + FACTORS1))
    out["beta_all"], out["r2_all"] = standardised_regression(df, all_factors, "monthly_sales")

    out["sales_effects"], out["r2_sales_within"] = within_effects(df, "monthly_sales", SALES_LEVERS)
    out["footfall_effects"], out["r2_foot_within"] = within_effects(df, "foot_traffic", FOOTFALL_LEVERS)
    out["levers"] = lever_table(df, out["sales_effects"])

    shared = ["promo_days", "is_holiday_month"]
    out["shared_table"] = pd.DataFrame({"Stage 1: footfall": out["beta1"][shared],
                                        "Stage 2: sales": out["beta2"][shared],
                                        "Stage 2 + ad spend": out["beta_ad"][shared]}).round(3)
    return out


def add_expected(df: pd.DataFrame) -> pd.DataFrame:
    """Expected footfall / sales from the pooled real-unit models (notebook D1)."""
    d = df.copy()
    d["expected_footfall"] = raw_fit(d, FACTORS1, "foot_traffic")[1]
    d["expected_sales"] = raw_fit(d, FACTORS2, "monthly_sales")[1]
    return d
