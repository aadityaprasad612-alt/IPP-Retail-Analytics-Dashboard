# Retail Analytics Dashboard (IPP Project)

Streamlit + Plotly dashboard for footfall and sales analysis of 40 stores x 24 months.

## Files
- `app.py` - dashboard (9 sections, sidebar filters)
- `analysis.py` - loading, cleaning, regressions
- `store_analysis.py` - `Store` class, scorecard, expected vs actual
- `budget_simulator.py` - `plan_budget()` and budget curve
- `requirements.txt`

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data
The app loads `retail_sales_demo_data.csv` from the same folder if present, otherwise from the GitHub raw URL
in `analysis.py` (`DATA_URL`). For a reliable deployment, commit the CSV next to `app.py`.

## Deploy on Streamlit Community Cloud
1. Push all files to a GitHub repository.
2. Go to https://share.streamlit.io, click **New app**, pick the repo and branch.
3. Set **Main file path** to `app.py` and click **Deploy**.
