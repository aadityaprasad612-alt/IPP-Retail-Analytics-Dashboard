"""budget_simulator.py - plan_budget() logic from notebook D6, returning data instead of printing."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd

STEP = {"ad_spend": 1, "promo_days": 1, "staff_count": 1, "inventory_fill_rate": 0.01}
PLAN_LEVERS = ["ad_spend", "promo_days", "staff_count", "inventory_fill_rate"]


def plan_budget(store, budget: float, effects: pd.Series, goal: str = "footfall",
                unit_costs: Optional[Dict[str, float]] = None) -> Tuple[pd.DataFrame, float, List[str]]:
    """Split an extra monthly budget across levers for ONE store, best value for money first.

    Returns (plan table, unspent budget, list of levers not included and why).
    """
    unit_costs = {k: v for k, v in (unit_costs or {}).items() if v and v > 0}
    costs = {"ad_spend": 1.0}            # ad spend is money itself
    costs.update(unit_costs)

    options, skipped = [], []
    for lever in PLAN_LEVERS:
        if lever not in effects.index:
            continue
        if lever not in costs:
            skipped.append(f"{lever}: cost not provided")
            continue
        gain_per_step = effects[lever] * STEP[lever]
        if gain_per_step <= 0:
            skipped.append(f"{lever}: no positive effect in the data")
            continue
        room = (store.rows[lever].max() - store.rows[lever].mean()) / STEP[lever]  # stay inside what the store has tried
        options.append({"lever": lever, "gain_per_step": gain_per_step, "cost_per_step": costs[lever],
                        "room_steps": room, "gain_per_cost": gain_per_step / costs[lever]})

    plan, remaining = [], float(budget)
    if options:
        opts = pd.DataFrame(options).sort_values("gain_per_cost", ascending=False)
        for _, o in opts.iterrows():
            steps = max(0.0, min(o["room_steps"], remaining / o["cost_per_step"]))
            spend = steps * o["cost_per_step"]
            remaining -= spend
            plan.append({"lever": o["lever"], "invest": round(spend, 0), "steps": round(steps, 2),
                         f"expected extra {goal}": round(steps * o["gain_per_step"], 1)})
    return pd.DataFrame(plan), remaining, skipped


def budget_curve(store, effects: pd.Series, goal: str, max_budget: float,
                 unit_costs: Optional[Dict[str, float]] = None, points: int = 21) -> pd.DataFrame:
    """Expected extra footfall/sales for a range of budgets (re-runs plan_budget at each level)."""
    col = f"expected extra {goal}"
    rows = []
    for i in range(points):
        b = max_budget * i / (points - 1)
        plan, _, _ = plan_budget(store, b, effects, goal, unit_costs)
        rows.append({"budget": b, "expected_gain": float(plan[col].sum()) if len(plan) else 0.0})
    return pd.DataFrame(rows)
