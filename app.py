"""app.py - Retail analytics dashboard (Streamlit + Plotly).

Run:  streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analysis as an
from budget_simulator import budget_curve, plan_budget
from store_analysis import all_stores_table, build_store_table, get_store, verdict

st.set_page_config(page_title="Retail Analytics Dashboard", page_icon="📊", layout="wide")

BLUE, ORANGE, GREEN, RED, GREY = "#2a6fdb", "#f28e2b", "#2ca58d", "#d1495b", "#8a94a6"
VERDICT_COLORS = {"Above expected": GREEN, "In line": GREY, "Below expected": RED}
st.markdown("""<style>
div[data-testid="stMetric"]{background:rgba(120,130,150,.08);border-radius:10px;padding:12px 16px;}
.block-container{padding-top:2rem;}
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- data
@st.cache_data(show_spinner="Loading data...")
def get_data(source: str | None = None):
    raw = an.load_raw(source)
    df, report = an.clean_data(raw)
    return df, report


@st.cache_data(show_spinner="Fitting models...")
def get_models(df: pd.DataFrame):
    models = an.run_models(df)
    dfx = an.add_expected(df)
    table = build_store_table(dfx)
    return models, dfx, table


def fmt(x, d=0):
    return f"{x:,.{d}f}"


def style_fig(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), template="plotly_white",
                      legend=dict(orientation="h", y=-0.2))
    return fig


def bar_importance(table: pd.DataFrame, title: str):
    t = table.reset_index()
    t["factor_label"] = t["factor"].map(an.label)
    t["sign"] = np.where(t["coefficient"] >= 0, "Positive", "Negative")
    fig = px.bar(t.sort_values("importance_%"), x="importance_%", y="factor_label", orientation="h",
                 color="sign", color_discrete_map={"Positive": BLUE, "Negative": RED},
                 hover_data={"coefficient": True, "rank": True}, title=title,
                 labels={"importance_%": "Importance (%)", "factor_label": ""})
    return style_fig(fig)


def bar_corr(corr: pd.Series, title: str):
    c = corr.rename(index=an.label).reset_index()
    c.columns = ["factor", "r"]
    fig = px.bar(c.sort_values("r"), x="r", y="factor", orientation="h", title=title,
                 color="r", color_continuous_scale="RdBu", range_color=[-1, 1], labels={"r": "Correlation (r)", "factor": ""})
    fig.update_coloraxes(showscale=False)
    return style_fig(fig)


def heatmap(d: pd.DataFrame, cols: list, title: str):
    c = d[cols].corr().round(2)
    c.index = [an.label(i) for i in c.index]
    c.columns = [an.label(i) for i in c.columns]
    fig = px.imshow(c, text_auto=True, color_continuous_scale="RdBu", zmin=-1, zmax=1, title=title, aspect="auto")
    return style_fig(fig, 520)


def trendline(x, y):
    m, b = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    return xs, b + m * xs, m


# ----------------------------------------------------------------------------- load
try:
    df, quality = get_data()
    models, dfx, store_table = get_models(df)
except Exception as err:
    st.error(f"Could not load or process the data: {err}")
    st.info("Place `retail_sales_demo_data.csv` next to app.py, or check the internet connection to GitHub.")
    st.stop()

# ----------------------------------------------------------------------------- sidebar
PAGES = ["Executive Overview", "Footfall Intelligence", "Sales Intelligence", "Sales vs Footfall",
         "Store Intelligence", "Expected vs Actual", "Store Comparison", "Budget Simulator",
         "Managerial Insights"]
st.sidebar.title("📊 Retail Analytics")
page = st.sidebar.radio("Section", PAGES)
st.sidebar.markdown("---")
st.sidebar.subheader("Filters")
years = sorted(dfx["year"].unique())
sel_years = st.sidebar.multiselect("Year", years, default=years)
sel_months = st.sidebar.multiselect("Month", list(range(1, 13)), default=list(range(1, 13)))
holiday_opt = st.sidebar.radio("Period type", ["All months", "Holiday months only", "Non-holiday months only"])
store_ids = sorted(dfx["store_id"].unique())
sel_stores = st.sidebar.multiselect("Stores (empty = all)", store_ids, default=[])
tolerance = st.sidebar.slider("Expected-vs-actual tolerance (%)", 1, 30, 10,
                              help="Notebook rule: more than 10% above/below expected.")
st.sidebar.caption("Filters apply to trends, KPIs and scatter plots. Regression models and store scorecards "
                   "are fitted on the full cleaned dataset, as in the notebook.")

f = dfx[dfx["year"].isin(sel_years) & dfx["month"].isin(sel_months)]
if holiday_opt.startswith("Holiday"):
    f = f[f["is_holiday_month"] == 1]
elif holiday_opt.startswith("Non"):
    f = f[f["is_holiday_month"] == 0]
if sel_stores:
    f = f[f["store_id"].isin(sel_stores)]
needs_filter = page in ("Executive Overview", "Sales vs Footfall")
if needs_filter and f.empty:
    st.warning("No data matches the current filters.")
    st.stop()

if "month_index" not in dfx.columns:
    dfx["month_index"] = (dfx["year"] - dfx["year"].min()) * 12 + dfx["month"]

stores_tbl = all_stores_table(store_table, tolerance)
st.title(page)


# ----------------------------------------------------------------------------- pages
def page_overview():
    tot_sales, tot_foot = f["monthly_sales"].sum(), f["foot_traffic"].sum()
    c = st.columns(5)
    c[0].metric("Total sales", fmt(tot_sales))
    c[1].metric("Total footfall", fmt(tot_foot))
    c[2].metric("Sales per visitor", fmt(tot_sales / tot_foot, 2))
    c[3].metric("Avg monthly sales / store", fmt(f["monthly_sales"].mean()))
    c[4].metric("Avg monthly footfall / store", fmt(f["foot_traffic"].mean()))
    st.caption(f"{len(f):,} store-months | {f['store_id'].nunique()} stores | "
               f"data quality: {quality['missing_ad_spend']} missing ad_spend filled, "
               f"{quality['impossible_footfall_fixed']} impossible footfall fixed")

    tr = f.groupby("month_index").agg(sales=("monthly_sales", "mean"), footfall=("foot_traffic", "mean"),
                                      sales_total=("monthly_sales", "sum"), foot_total=("foot_traffic", "sum"),
                                      holiday=("is_holiday_month", "max")).reset_index()
    tr["spv"] = tr["sales_total"] / tr["foot_total"]
    a, b = st.columns(2)
    fig = px.line(tr, x="month_index", y="sales", markers=True, title="Average monthly sales per store",
                  labels={"month_index": "Month index", "sales": "Sales"})
    fig.update_traces(line_color=BLUE)
    a.plotly_chart(style_fig(fig), use_container_width=True)
    fig = px.line(tr, x="month_index", y="footfall", markers=True, title="Average footfall per store",
                  labels={"month_index": "Month index", "footfall": "Visitors"})
    fig.update_traces(line_color=ORANGE)
    b.plotly_chart(style_fig(fig), use_container_width=True)

    a, b = st.columns(2)
    fig = px.line(tr, x="month_index", y="spv", markers=True, title="Sales per visitor (chain)",
                  labels={"month_index": "Month index", "spv": "Sales / visitor"})
    fig.update_traces(line_color=GREEN)
    a.plotly_chart(style_fig(fig), use_container_width=True)
    seas = f.groupby("month")[["monthly_sales", "foot_traffic"]].mean().reset_index()
    fig = go.Figure()
    fig.add_bar(x=seas["month"], y=seas["monthly_sales"], name="Avg sales", marker_color=BLUE)
    fig.add_scatter(x=seas["month"], y=seas["foot_traffic"], name="Avg footfall", yaxis="y2",
                    mode="lines+markers", line=dict(color=ORANGE))
    fig.update_layout(title="Seasonality by calendar month", xaxis=dict(dtick=1, title="Month"),
                      yaxis=dict(title="Sales"), yaxis2=dict(title="Footfall", overlaying="y", side="right"))
    b.plotly_chart(style_fig(fig), use_container_width=True)

    top = f.groupby("store_id")["monthly_sales"].mean().sort_values(ascending=False)
    k = min(10, len(top))
    a, b = st.columns(2)
    t1 = top.head(k).reset_index().astype({"store_id": str})
    a.plotly_chart(style_fig(px.bar(t1, x="store_id", y="monthly_sales", title=f"Top {k} stores by avg sales",
                                    color_discrete_sequence=[BLUE], labels={"store_id": "Store", "monthly_sales": "Sales"})),
                   use_container_width=True)
    t2 = top.tail(k).reset_index().astype({"store_id": str})
    b.plotly_chart(style_fig(px.bar(t2, x="store_id", y="monthly_sales", title=f"Bottom {k} stores by avg sales",
                                    color_discrete_sequence=[RED], labels={"store_id": "Store", "monthly_sales": "Sales"})),
                   use_container_width=True)

    with st.expander("Holiday vs non-holiday averages"):
        h = f.groupby("is_holiday_month")[["foot_traffic", "monthly_sales", "ad_spend", "promo_days", "avg_price"]].mean().round(1)
        h.index = h.index.map({0: "Non-holiday", 1: "Holiday"})
        st.dataframe(h, use_container_width=True)


def page_footfall():
    r1, res1 = models["r2_1"], models["result1"]
    c = st.columns(3)
    c[0].metric("Model R-squared", f"{r1:.2f}", help="Share of footfall variation explained by Stage 1 factors")
    c[1].metric("Top factor", an.label(res1.index[0]))
    c[2].metric("Visitors per 1,000 ad spend", fmt(models["effect1"]["ad_spend"] * 1000))
    st.caption(f"Stage 1 factors explain about {r1 * 100:.0f}% of the variation in footfall.")

    t1, t2, t3, t4 = st.tabs(["Factor ranking", "Correlation", "Advertising impact", "Actual vs predicted"])
    with t1:
        a, b = st.columns([3, 2])
        a.plotly_chart(bar_importance(res1, "Stage 1 factor ranking (standardised regression)"), use_container_width=True)
        eff = models["effect1"].rename(index=an.label).round(3).rename("extra visitors per +1 unit").to_frame()
        b.markdown("**Real-unit effect (others fixed)**")
        b.dataframe(eff, use_container_width=True)
        st.dataframe(res1.rename(index=an.label), use_container_width=True)
        st.caption("Parking spaces overlaps with store size (r = 0.98): read it as 'bigger store'.")
    with t2:
        a, b = st.columns(2)
        a.plotly_chart(bar_corr(an.correlations(dfx, an.FACTORS1, "foot_traffic"), "Correlation with footfall"),
                       use_container_width=True)
        b.plotly_chart(heatmap(dfx, ["foot_traffic"] + an.FACTORS1, "Correlation heatmap (Stage 1)"),
                       use_container_width=True)
    with t3:
        sample = dfx
        fig = px.scatter(sample, x="ad_spend", y="foot_traffic", opacity=0.45,
                         color=sample["is_holiday_month"].map({0: "Non-holiday", 1: "Holiday"}),
                         color_discrete_map={"Non-holiday": BLUE, "Holiday": ORANGE},
                         title="Ad spend vs footfall", labels={"ad_spend": "Ad spend", "foot_traffic": "Footfall", "color": ""})
        xs, ys, _ = trendline(sample["ad_spend"], sample["foot_traffic"])
        fig.add_scatter(x=xs, y=ys, mode="lines", name="Linear fit", line=dict(color="black", dash="dash"))
        st.plotly_chart(style_fig(fig, 420), use_container_width=True)
        ff = models["footfall_effects"]
        st.info(f"Chain-wide model: about {models['effect1']['ad_spend'] * 1000:,.0f} extra visitors per 1,000 ad spend. "
                f"Inside a single store (store averages removed): about {ff['ad_spend'] * 1000:,.0f}. "
                f"Promo day: about {models['effect1']['promo_days']:,.0f} extra visitors (chain-wide).")
    with t4:
        fig = px.scatter(dfx, x="expected_footfall", y="foot_traffic", opacity=0.45,
                         title="Actual vs predicted footfall (store-months)",
                         labels={"expected_footfall": "Predicted footfall", "foot_traffic": "Actual footfall"})
        lo, hi = dfx[["expected_footfall", "foot_traffic"]].min().min(), dfx[["expected_footfall", "foot_traffic"]].max().max()
        fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect fit", line=dict(color="black", dash="dash"))
        st.plotly_chart(style_fig(fig, 450), use_container_width=True)
        m = dfx.groupby("month_index")[["foot_traffic", "expected_footfall"]].mean().reset_index() if "month_index" in dfx.columns else None
        if m is not None:
            fig = px.line(m, x="month_index", y=["foot_traffic", "expected_footfall"], title="Average actual vs predicted over time",
                          labels={"value": "Footfall", "month_index": "Month index", "variable": ""})
            st.plotly_chart(style_fig(fig), use_container_width=True)


def page_sales():
    r2, r2a = models["r2_2"], models["r2_ad"]
    c = st.columns(4)
    c[0].metric("Stage 2 R-squared", f"{r2:.2f}")
    c[1].metric("With ad spend added", f"{r2a:.2f}", delta=f"+{(r2a - r2) * 100:.0f} pts")
    c[2].metric("With all Stage 1 factors", f"{models['r2_all']:.2f}")
    c[3].metric("Top factor (Stage 2)", an.label(models["result2"].index[0]))

    t1, t2, t3, t4 = st.tabs(["Factor ranking", "Model comparison", "Correlation", "Manager levers"])
    with t1:
        a, b = st.columns(2)
        a.plotly_chart(bar_importance(models["result2"], f"Stage 2 ranking (R-squared = {r2:.2f})"), use_container_width=True)
        b.plotly_chart(bar_importance(models["check_ad"], f"With ad spend (R-squared = {r2a:.2f})"), use_container_width=True)
        eff = models["effect2"].rename(index=an.label).round(3).rename("extra sales per +1 unit").to_frame()
        st.markdown("**Real-unit effect, Stage 2 model (others fixed)**")
        st.dataframe(eff, use_container_width=True)
    with t2:
        comp = pd.DataFrame({"Model": ["Stage 2 factors", "Stage 2 + ad spend", "Stage 2 + all Stage 1 factors"],
                             "R-squared": [r2, r2a, models["r2_all"]]})
        fig = px.bar(comp, x="Model", y="R-squared", text=comp["R-squared"].map(lambda v: f"{v:.0%}"),
                     color="Model", color_discrete_sequence=[GREY, BLUE, GREEN], title="Share of sales variation explained")
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig), use_container_width=True)
        st.markdown("**Shared factors across models (standardised coefficients)**")
        st.dataframe(models["shared_table"].rename(index=an.label), use_container_width=True)
        st.caption("Footfall itself is deliberately not in any sales model (as decided in the project).")
    with t3:
        a, b = st.columns(2)
        a.plotly_chart(bar_corr(an.correlations(dfx, an.FACTORS2, "monthly_sales"), "Correlation with monthly sales"),
                       use_container_width=True)
        b.plotly_chart(heatmap(dfx, ["monthly_sales"] + an.FACTORS2 + ["store_size_sqft"], "Correlation heatmap (Stage 2)"),
                       use_container_width=True)
    with t4:
        lv = models["levers"].copy()
        lv_plot = lv.reset_index()
        lv_plot["lever"] = lv_plot["lever"].map(an.label)
        fig = px.bar(lv_plot.sort_values("extra_sales_per_month"), x="extra_sales_per_month", y="lever", orientation="h",
                     hover_data=["change", "typical_swing_in_a_store"], title="Effect of one step inside a store (ad spend controlled)",
                     color="extra_sales_per_month", color_continuous_scale="RdBu", color_continuous_midpoint=0,
                     labels={"extra_sales_per_month": "Extra monthly sales", "lever": ""})
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(style_fig(fig), use_container_width=True)
        st.dataframe(lv, use_container_width=True)
        st.caption("Costs of price changes, promo days and staff are not in the data. Small effects are read as no detectable effect "
                   "(no significance tests were run).")


def page_link():
    link = an.footfall_sales_link(f) if f["is_holiday_month"].nunique() == 2 else None
    ov = f["foot_traffic"].corr(f["monthly_sales"])
    xs, ys, slope = trendline(f["foot_traffic"], f["monthly_sales"])
    c = st.columns(3)
    c[0].metric("Correlation (filtered data)", f"{ov:.2f}")
    c[1].metric("Sales per 1,000 extra visitors", fmt(slope * 1000))
    c[2].metric("Avg sales per visitor", fmt(f["monthly_sales"].sum() / f["foot_traffic"].sum(), 2))

    a, b = st.columns(2)
    fig = px.scatter(f, x="foot_traffic", y="monthly_sales", opacity=0.45,
                     color=f["is_holiday_month"].map({0: "Non-holiday", 1: "Holiday"}),
                     color_discrete_map={"Non-holiday": BLUE, "Holiday": ORANGE}, hover_data=["store_id", "month"],
                     title="Footfall vs sales", labels={"foot_traffic": "Footfall", "monthly_sales": "Sales", "color": ""})
    fig.add_scatter(x=xs, y=ys, mode="lines", name="Linear fit", line=dict(color="black", dash="dash"))
    a.plotly_chart(style_fig(fig, 420), use_container_width=True)
    avg = f.groupby("store_id")[["foot_traffic", "monthly_sales"]].mean().reset_index()
    fig = px.scatter(avg, x="foot_traffic", y="monthly_sales", text="store_id", title="Store averages",
                     labels={"foot_traffic": "Avg footfall", "monthly_sales": "Avg sales"})
    fig.update_traces(textposition="top center", marker=dict(color=BLUE, size=8))
    b.plotly_chart(style_fig(fig, 420), use_container_width=True)

    a, b = st.columns(2)
    hv = f.groupby("is_holiday_month").agg(footfall=("foot_traffic", "mean"), sales=("monthly_sales", "mean"),
                                           fs=("foot_traffic", "sum"), ss=("monthly_sales", "sum")).reset_index()
    hv["Period"] = hv["is_holiday_month"].map({0: "Non-holiday", 1: "Holiday"})
    hv["Sales per visitor"] = hv["ss"] / hv["fs"]
    m = hv.melt(id_vars="Period", value_vars=["footfall", "sales"], var_name="Measure", value_name="Average")
    fig = px.bar(m, x="Measure", y="Average", color="Period", barmode="group", title="Holiday vs non-holiday (averages)",
                 color_discrete_map={"Non-holiday": BLUE, "Holiday": ORANGE})
    a.plotly_chart(style_fig(fig), use_container_width=True)
    fig = px.bar(hv, x="Period", y="Sales per visitor", color="Period", title="Sales per visitor",
                 color_discrete_map={"Non-holiday": BLUE, "Holiday": ORANGE})
    fig.update_layout(showlegend=False)
    b.plotly_chart(style_fig(fig), use_container_width=True)

    st.markdown("**Correlation between footfall and sales (filtered data)**")
    if link is not None and len(f) > 10:
        st.dataframe(link.rename("r").to_frame(), use_container_width=True)
    else:
        st.caption("Select both holiday and non-holiday months to see the split.")
    full = an.footfall_sales_link(dfx)
    st.markdown("**Same table on the full dataset**")
    st.dataframe(full.rename("r").to_frame(), use_container_width=True)
    st.caption("Correlation shows a relationship, not that one thing causes the other.")

    g = f.groupby("store_id")[["monthly_sales", "foot_traffic"]].sum()
    spv = (g["monthly_sales"] / g["foot_traffic"]).rename("spv").reset_index().sort_values("spv", ascending=False)
    spv["store_id"] = spv["store_id"].astype(str)
    fig = px.bar(spv, x="store_id", y="spv", title="Sales per visitor by store", color_discrete_sequence=[GREEN],
                 labels={"store_id": "Store", "spv": "Sales / visitor"})
    fig.add_hline(y=f["monthly_sales"].sum() / f["foot_traffic"].sum(), line_dash="dash", annotation_text="average")
    st.plotly_chart(style_fig(fig), use_container_width=True)


def select_store(key: str):
    sid = st.selectbox("Select store", store_ids, index=store_ids.index(15) if 15 in store_ids else 0, key=key)
    store = get_store(sid, dfx, store_table)
    if store is None:
        st.error(f"Store {sid} not found.")
        st.stop()
    return store


def page_store():
    store = select_store("store_intel")
    s = store.summary()
    c = st.columns(4)
    c[0].metric("Avg footfall / month", fmt(s["footfall"]), f"rank {s['footfall_rank']} of {s['n_stores']}", delta_color="off")
    c[1].metric("Avg sales / month", fmt(s["sales"]), f"rank {s['sales_rank']} of {s['n_stores']}", delta_color="off")
    c[2].metric("Sales per visitor", fmt(s["spv"], 2), f"chain {s['chain_spv']:.2f}", delta_color="off")
    c[3].metric("Sales/visitor rank", f"{s['spv_rank']} of {s['n_stores']}")

    sc = store.scorecard()
    a, b = st.columns([3, 2])
    a.markdown("**Scorecard: store vs chain**")
    a.dataframe(sc.rename(index=an.label), use_container_width=True, height=440)
    p = sc.reset_index().rename(columns={"index": "factor"})
    p["factor"] = p["factor"].map(an.label)
    fig = px.bar(p.sort_values("percentile"), x="percentile", y="factor", orientation="h", color="position",
                 color_discrete_map={"High (top quarter)": GREEN, "Middle": GREY, "Low (bottom quarter)": RED},
                 title="Percentile among stores", range_x=[0, 100], labels={"factor": ""})
    b.plotly_chart(style_fig(fig, 440), use_container_width=True)
    st.caption("A high or low position describes where the store stands; it is not a verdict on what to change.")

    r = store.rows.sort_values("month_index") if "month_index" in store.rows.columns else store.rows
    xcol = "month_index" if "month_index" in r.columns else "month"
    a, b = st.columns(2)
    fig = go.Figure()
    fig.add_scatter(x=r[xcol], y=r["monthly_sales"], name="Actual sales", mode="lines+markers", line=dict(color=BLUE))
    fig.add_scatter(x=r[xcol], y=r["expected_sales"], name="Expected sales", mode="lines", line=dict(color=GREY, dash="dash"))
    fig.update_layout(title="Monthly sales: actual vs expected", xaxis_title="Month index")
    a.plotly_chart(style_fig(fig), use_container_width=True)
    fig = go.Figure()
    fig.add_scatter(x=r[xcol], y=r["foot_traffic"], name="Actual footfall", mode="lines+markers", line=dict(color=ORANGE))
    fig.add_scatter(x=r[xcol], y=r["expected_footfall"], name="Expected footfall", mode="lines", line=dict(color=GREY, dash="dash"))
    fig.update_layout(title="Monthly footfall: actual vs expected", xaxis_title="Month index")
    b.plotly_chart(style_fig(fig), use_container_width=True)


def page_expected():
    store = select_store("store_exp")
    ev = store.expected_vs_actual(tolerance)
    c = st.columns(2)
    for col, (name, row) in zip(c, ev.iterrows()):
        col.metric(f"{name}: actual vs expected", fmt(row["actual"]), f"{row['gap_%']:+.1f}% vs expected ({fmt(row['expected'])})",
                   delta_color="normal")
        col.markdown(f"**Verdict:** :{'green' if row['verdict'] == 'Above expected' else 'red' if row['verdict'] == 'Below expected' else 'gray'}[{row['verdict']}]")
    st.dataframe(ev, use_container_width=True)
    st.caption(f"Rule: more than {tolerance}% above expected = Above, more than {tolerance}% below = Below, otherwise In line. "
               "The gap is what the models cannot see; it is not proof of good or poor management.")

    counts = stores_tbl["sales_verdict"].value_counts().reindex(["Below expected", "In line", "Above expected"]).fillna(0).reset_index()
    counts.columns = ["verdict", "stores"]
    a, b = st.columns(2)
    fig = px.bar(counts, x="verdict", y="stores", color="verdict", color_discrete_map=VERDICT_COLORS,
                 title="All stores: sales vs expected", text="stores")
    fig.update_layout(showlegend=False)
    a.plotly_chart(style_fig(fig), use_container_width=True)
    fig = px.scatter(stores_tbl.reset_index(), x="footfall_gap_%", y="sales_gap_%", color="sales_verdict", text="store_id",
                     color_discrete_map=VERDICT_COLORS, title="Footfall gap vs sales gap (all stores)",
                     labels={"footfall_gap_%": "Footfall gap %", "sales_gap_%": "Sales gap %"})
    fig.update_traces(textposition="top center")
    fig.add_vline(x=0, line_dash="dot")
    fig.add_hline(y=0, line_dash="dot")
    sel = stores_tbl.loc[[store.store_id]]
    fig.add_scatter(x=sel["footfall_gap_%"], y=sel["sales_gap_%"], mode="markers", name=f"Store {store.store_id}",
                    marker=dict(size=16, symbol="circle-open", color="black", line=dict(width=3)))
    b.plotly_chart(style_fig(fig), use_container_width=True)


def page_compare():
    metric_map = {"Average sales": "monthly_sales", "Average footfall": "foot_traffic", "Sales per visitor": "sales_per_visitor",
                  "Sales gap % vs expected": "sales_gap_%", "Footfall gap % vs expected": "footfall_gap_%"}
    m = st.selectbox("Rank stores by", list(metric_map))
    col = metric_map[m]
    t = stores_tbl.reset_index().sort_values(col, ascending=False)
    t["store_label"] = t["store_id"].astype(str)
    fig = px.bar(t, x="store_label", y=col, title=f"All stores ranked by {m.lower()}", labels={"store_label": "Store", col: m},
                 color="sales_verdict" if "Sales" in m or "visitor" in m else "footfall_verdict", color_discrete_map=VERDICT_COLORS)
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=t["store_label"])
    st.plotly_chart(style_fig(fig, 430), use_container_width=True)

    picks = st.multiselect("Compare selected stores", store_ids, default=store_ids[:3])
    if picks:
        sub = stores_tbl.loc[picks]
        cols = ["foot_traffic", "monthly_sales", "sales_per_visitor", "ad_spend", "promo_days", "avg_price"]
        norm = (sub[cols] / stores_tbl[cols].mean()).reset_index().melt(id_vars="store_id", var_name="metric", value_name="vs chain avg")
        norm["metric"] = norm["metric"].map(an.label).fillna(norm["metric"].replace({"sales_per_visitor": "Sales per visitor"}))
        norm["store_id"] = norm["store_id"].astype(str)
        fig = px.bar(norm, x="metric", y="vs chain avg", color="store_id", barmode="group", title="Selected stores relative to chain average (1.0 = average)")
        fig.add_hline(y=1, line_dash="dash")
        st.plotly_chart(style_fig(fig, 400), use_container_width=True)

    show = stores_tbl[["foot_traffic", "footfall_rank", "monthly_sales", "sales_rank", "sales_per_visitor", "spv_rank",
                       "footfall_gap_%", "footfall_verdict", "sales_gap_%", "sales_verdict"]].round(2)
    st.dataframe(show.sort_values("sales_rank"), use_container_width=True, height=500)
    st.download_button("Download store table (CSV)", show.to_csv().encode("utf-8"), "store_comparison.csv", "text/csv")


def page_budget():
    store = select_store("store_budget")
    c = st.columns(3)
    goal = c[0].radio("Goal", ["footfall", "sales"], horizontal=True)
    budget = c[1].number_input("Extra monthly budget", min_value=0.0, value=5000.0, step=500.0)
    c[2].caption("Budget = extra monthly spend above what the store spends now. Ad spend costs 1 per unit; other lever costs are not in the data.")
    st.markdown("**Optional unit costs** (cost of ONE step; leave 0 to exclude the lever)")
    u = st.columns(3)
    unit_costs = {"promo_days": u[0].number_input("Cost of one promotion day", 0.0, step=100.0),
                  "staff_count": u[1].number_input("Cost of one extra staff member", 0.0, step=100.0),
                  "inventory_fill_rate": u[2].number_input("Cost of +1 pt stock availability", 0.0, step=100.0)}
    effects = models["footfall_effects"] if goal == "footfall" else models["sales_effects"]
    try:
        plan, remaining, skipped = plan_budget(store, budget, effects, goal, unit_costs)
    except Exception as err:
        st.error(f"Could not build the plan: {err}")
        return
    gain_col = f"expected extra {goal}"
    k = st.columns(3)
    k[0].metric("Budget invested", fmt(budget - remaining))
    k[1].metric(f"Expected extra {goal} / month", fmt(plan[gain_col].sum(), 1) if len(plan) else "0")
    k[2].metric("Unspent budget", fmt(remaining))
    if len(plan):
        a, b = st.columns(2)
        a.dataframe(plan, use_container_width=True, hide_index=True)
        fig = px.pie(plan[plan["invest"] > 0], names="lever", values="invest", title="Budget allocation", hole=0.45)
        b.plotly_chart(style_fig(fig, 320), use_container_width=True)
    else:
        st.warning("No lever could be planned.")
    for note in skipped:
        st.caption(f"Not included - {note}")
    curve = budget_curve(store, effects, goal, max(budget, 1.0) * 1.5, unit_costs)
    fig = px.line(curve, x="budget", y="expected_gain", markers=True, title=f"Expected extra {goal} by budget level",
                  labels={"budget": "Extra monthly budget", "expected_gain": f"Expected extra {goal}"})
    fig.add_vline(x=budget, line_dash="dash", annotation_text="chosen budget")
    st.plotly_chart(style_fig(fig), use_container_width=True)
    st.caption("Expected gains assume a straight line, chain-wide inside-store averages, and are capped at the store's own observed range. "
               "Units of ad spend and sales are not stated, so no return on investment is calculated.")


def page_insights():
    r1, r2, r2a = models["r2_1"], models["r2_2"], models["r2_ad"]
    res1, res2, resa = models["result1"], models["result2"], models["check_ad"]
    link = an.footfall_sales_link(dfx)
    lv = models["levers"]
    below = int((stores_tbl["sales_verdict"] == "Below expected").sum())
    inline = int((stores_tbl["sales_verdict"] == "In line").sum())
    above = int((stores_tbl["sales_verdict"] == "Above expected").sum())
    h = dfx.groupby("is_holiday_month")[["foot_traffic", "monthly_sales"]].mean()
    ff_up = (h.loc[1, "foot_traffic"] / h.loc[0, "foot_traffic"] - 1) * 100
    s_up = (h.loc[1, "monthly_sales"] / h.loc[0, "monthly_sales"] - 1) * 100
    top1 = [an.label(i) for i in res1.index[:3]]
    top2 = [an.label(i) for i in res2.index[:3]]
    topa = [an.label(i) for i in resa.index[:3]]
    gap_s = stores_tbl["sales_gap_%"]
    gap_f = stores_tbl["footfall_gap_%"]

    st.subheader("Key findings")
    st.markdown(f"""
1. **Footfall drivers:** Stage 1 factors explain {r1 * 100:.0f}% of footfall. Top three: {', '.join(top1)}. About {models['effect1']['ad_spend'] * 1000:,.0f} extra visitors per 1,000 ad spend.
2. **Sales drivers:** Stage 2 factors explain {r2 * 100:.0f}% of sales (top three: {', '.join(top2)}). Adding ad spend lifts this to {r2a * 100:.0f}% and the top three become {', '.join(topa)}.
3. **Footfall and sales:** correlation is {link.iloc[0]:.2f} overall and {link.iloc[1]:.2f} in non-holiday months, so the link is not only a holiday effect; across store averages it is {link.iloc[3]:.2f}.
4. **Holiday effect:** holiday months have {ff_up:.0f}% more footfall and {s_up:.0f}% more sales than other months.
5. **Store performance:** with the {tolerance}% rule, {below} stores are below, {inline} in line and {above} above expected sales. Sales gaps range from {gap_s.min():.0f}% to {gap_s.max():+.0f}% while footfall gaps range from {gap_f.min():.0f}% to {gap_f.max():+.0f}%, so the biggest differences arise after customers arrive.
""")
    st.subheader("Recommendations (what a manager can change)")
    ad = lv.loc["ad_spend", "extra_sales_per_month"]
    pr = lv.loc["avg_price", "extra_sales_per_month"]
    pd_ = lv.loc["promo_days", "extra_sales_per_month"]
    st.markdown(f"""
- **Advertising** is the lever with the strongest and most consistent link: +1,000 ad spend goes with about {ad:,.0f} more monthly sales inside a store and about {models['footfall_effects']['ad_spend'] * 1000:,.0f} more visitors. Units are not stated, so payback cannot be judged.
- **Promotion days** are a modest lever: about {pd_:,.0f} more monthly sales per extra day once ad spend is controlled; promotion day cost is not in the data (enter it in the Budget Simulator).
- **Average price:** +1 goes with about {pr:,.0f} more monthly sales, but this is a large step relative to typical in-store swings, and sales are revenue, so read it as "no sign of a sales loss", not "raise prices".
- **Staffing and stock availability** show no detectable effect on sales in this data; do not raise them for sales reasons alone.
- **Stores below expected sales** (see Store Comparison) are questions to investigate, for example product mix or local conditions, not proof of poor management.
""")
    st.subheader("Limitations")
    st.markdown("""
- Results show relationships, not proven causes; no significance tests were run.
- Only 40 stores: store-level factors (income, size, parking, competitors, age) are the least certain findings.
- Staff count and store size carry almost the same signal (r = 0.95); parking and store size too (r = 0.98).
- Footfall is not in the sales model; ad spend is not in the base Stage 2 model, so factor effects include effects that run through them.
- Expected-vs-actual gaps are what the models cannot see; budget plans assume straight-line, chain-wide effects.
""")
    st.subheader("Data quality")
    st.json(quality)


{"Executive Overview": page_overview, "Footfall Intelligence": page_footfall, "Sales Intelligence": page_sales,
 "Sales vs Footfall": page_link, "Store Intelligence": page_store, "Expected vs Actual": page_expected,
 "Store Comparison": page_compare, "Budget Simulator": page_budget, "Managerial Insights": page_insights}[page]()
