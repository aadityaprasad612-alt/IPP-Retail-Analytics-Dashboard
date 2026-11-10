"""store_analysis.py - store table, Store class, scorecard and expected-versus-actual (notebook D1-D7)."""
from __future__ import annotations

from typing import Optional

import pandas as pd

USED_IN = {"ad_spend": "Footfall", "promo_days": "Both", "competitors_within_3km": "Footfall",
           "parking_spaces": "Footfall", "store_age_years": "Footfall", "avg_price": "Sales",
           "staff_count": "Sales", "inventory_fill_rate": "Sales", "returns_rate": "Sales",
           "local_income_index": "Sales", "store_size_sqft": "Context only"}


def position(percentile: float) -> str:
    if percentile >= 75:
        return "High (top quarter)"
    if percentile <= 25:
        return "Low (bottom quarter)"
    return "Middle"


def verdict(gap_pct: float, tolerance: float = 10) -> str:
    if gap_pct > tolerance:
        return "Above expected"
    if gap_pct < -tolerance:
        return "Below expected"
    return "In line"


def build_store_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per store: 24-month averages plus ranks. df must contain expected_footfall / expected_sales."""
    cols = list(USED_IN) + ["foot_traffic", "monthly_sales", "expected_footfall", "expected_sales"]
    t = df.groupby("store_id")[cols].mean()
    t["footfall_rank"] = t["foot_traffic"].rank(ascending=False, method="min").astype(int)
    t["sales_rank"] = t["monthly_sales"].rank(ascending=False, method="min").astype(int)
    t["sales_per_visitor"] = t["monthly_sales"] / t["foot_traffic"]
    t["spv_rank"] = t["sales_per_visitor"].rank(ascending=False, method="min").astype(int)
    return t


class Store:
    """One store: look it up, compare it with the chain."""

    def __init__(self, store_id: int, df: pd.DataFrame, store_table: pd.DataFrame):
        if store_id not in store_table.index:
            raise ValueError(f"Store {store_id} not found. Valid ids are "
                             f"{store_table.index.min()} to {store_table.index.max()}.")
        self.store_id = store_id
        self.store_table = store_table
        self.rows = df[df["store_id"] == store_id]
        self.avg = store_table.loc[store_id]

    def summary(self) -> dict:
        n = len(self.store_table)
        return {
            "months": len(self.rows),
            "footfall": self.avg["foot_traffic"], "footfall_rank": int(self.avg["footfall_rank"]),
            "sales": self.avg["monthly_sales"], "sales_rank": int(self.avg["sales_rank"]),
            "spv": self.avg["sales_per_visitor"], "spv_rank": int(self.avg["spv_rank"]),
            "chain_spv": self.store_table["sales_per_visitor"].mean(), "n_stores": n,
        }

    def scorecard(self) -> pd.DataFrame:
        cols = list(USED_IN)
        table = pd.DataFrame({
            "used_in": pd.Series(USED_IN),
            "this store": self.avg[cols],
            "chain average": self.store_table[cols].mean(),
            "percentile": [(self.store_table[c] <= self.avg[c]).mean() * 100 for c in cols],
        })
        table["position"] = [position(p) for p in table["percentile"]]
        return table.round(3)

    def expected_vs_actual(self, tolerance: float = 10) -> pd.DataFrame:
        rows = []
        for lab, actual, expected in [("Footfall", "foot_traffic", "expected_footfall"),
                                      ("Sales", "monthly_sales", "expected_sales")]:
            a, e = self.avg[actual], self.avg[expected]
            gap_pct = (a / e - 1) * 100
            rows.append({"measure": lab, "actual": round(a, 1), "expected": round(e, 1),
                         "gap": round(a - e, 1), "gap_%": round(gap_pct, 1),
                         "verdict": verdict(gap_pct, tolerance)})
        return pd.DataFrame(rows).set_index("measure")


def get_store(store_id, df: pd.DataFrame, store_table: pd.DataFrame) -> Optional[Store]:
    """Safe lookup: returns None instead of raising for a wrong id."""
    try:
        return Store(store_id, df, store_table)
    except ValueError:
        return None


def all_stores_table(store_table: pd.DataFrame, tolerance: float = 10) -> pd.DataFrame:
    """Notebook D7: every store with gap % and verdict for footfall and sales."""
    t = store_table.copy()
    t["footfall_gap_%"] = (t["foot_traffic"] / t["expected_footfall"] - 1) * 100
    t["sales_gap_%"] = (t["monthly_sales"] / t["expected_sales"] - 1) * 100
    t["footfall_verdict"] = t["footfall_gap_%"].apply(verdict, tolerance=tolerance)
    t["sales_verdict"] = t["sales_gap_%"].apply(verdict, tolerance=tolerance)
    return t
