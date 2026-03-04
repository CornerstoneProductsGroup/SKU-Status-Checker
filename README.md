# Home Depot Availability Checker (SerpApi) — Streamlit

## What this does
- Loads `SKU Map.xlsx` (sheet: `Depot`) with columns:
  `OMSID, Internet #, SKU #, UPC, GTIN, Vendor`
- Lets you select items (filter by vendor / ID), then checks availability via SerpApi.
- Outputs statuses: `AVAILABLE`, `OUT_OF_STOCK`, `NOT_AVAILABLE`, `OTHER`
- Exports results to CSV.

## Setup
1) Put your SerpApi key into Streamlit Secrets

### Local
Create `.streamlit/secrets.toml`:

```toml
SERPAPI_API_KEY="YOUR_KEY"
```

### Streamlit Cloud
App → Settings → Secrets, add the same line.

2) Install dependencies
```bash
pip install -r requirements.txt
```

3) Run
```bash
streamlit run app.py
```

## Notes
- Availability is location-dependent; enter a ZIP for better signals.
- This app uses caching to reduce API calls. Clear cache if needed in Streamlit menu.
