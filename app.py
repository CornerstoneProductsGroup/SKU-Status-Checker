import os

import time
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

APP_TITLE = "Home Depot SKU Availability Checker (SerpApi)"
SERPAPI_ENDPOINT = "https://serpapi.com/search.json"

# -----------------------------
# Helpers
# -----------------------------
def _safe_str(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()

def _digits_only(s: str) -> str:
    return re.sub(r"\D+", "", _safe_str(s))

def _as_int_str(x) -> str:
    """
    Excel often loads Internet # / UPC as floats.
    Convert to a clean integer-like string when possible.
    """
    s = _safe_str(x)
    if not s:
        return ""
    # handle numbers like 1.012973e+09 or 1012973000.0
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return _digits_only(s) or s

def _get_api_key() -> str:
    # Prefer Streamlit secrets; fall back to env var
    if "SERPAPI_API_KEY" in st.secrets:
        return st.secrets["SERPAPI_API_KEY"]
    return _safe_str(os.getenv("SERPAPI_API_KEY", ""))

def _http_get(params: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.get(SERPAPI_ENDPOINT, params=params, timeout=45)
    # Helpful error detail for debugging
    if r.status_code >= 400:
        raise RuntimeError(f"SerpApi HTTP {r.status_code}: {r.text[:500]}")
    return r.json()

@st.cache_data(show_spinner=False, ttl=60*60)
def hd_search_cached(q: str, delivery_zip: str = "", store_id: str = "") -> Dict[str, Any]:
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("Missing SERPAPI_API_KEY. Add it to Streamlit secrets.")
    params = {
        "api_key": api_key,
        "engine": "home_depot",
        "q": q,
        "country": "us",
    }
    if delivery_zip:
        params["delivery_zip"] = delivery_zip
    if store_id:
        params["store_id"] = store_id
    return _http_get(params)

@st.cache_data(show_spinner=False, ttl=60*60)
def hd_product_cached(product_id: str) -> Dict[str, Any]:
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("Missing SERPAPI_API_KEY. Add it to Streamlit secrets.")
    params = {
        "api_key": api_key,
        "engine": "home_depot_product",
        "product_id": product_id,
    }
    return _http_get(params)

def _find_product_candidates(search_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    # SerpApi engines vary; try common fields
    for key in ["products", "organic_results", "shopping_results", "results"]:
        v = search_json.get(key)
        if isinstance(v, list) and v:
            return v
    return []

def _extract_product_id_from_item(item: Dict[str, Any]) -> str:
    # Direct key
    for k in ["product_id", "productId", "id", "item_id", "internet_number"]:
        if k in item and _safe_str(item.get(k)):
            return _as_int_str(item.get(k))
    # Sometimes inside link URL (look for long digit runs)
    link = _safe_str(item.get("link") or item.get("url"))
    if link:
        m = re.findall(r"(\d{7,12})", link)
        if m:
            return m[-1]
    return ""

def resolve_product_id_from_query(q: str, delivery_zip: str = "", store_id: str = "") -> Tuple[str, Dict[str, Any]]:
    """
    Returns (product_id, picked_item_meta)
    """
    sjson = hd_search_cached(q=q, delivery_zip=delivery_zip, store_id=store_id)
    items = _find_product_candidates(sjson)
    if not items:
        return "", {}
    # Choose the first with a product_id
    for it in items[:10]:
        pid = _extract_product_id_from_item(it if isinstance(it, dict) else {})
        if pid:
            return pid, it if isinstance(it, dict) else {}
    return "", items[0] if isinstance(items[0], dict) else {}

def _flatten_text(obj: Any, limit_chars: int = 20000) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False)
    except Exception:
        s = _safe_str(obj)
    if len(s) > limit_chars:
        s = s[:limit_chars]
    return s.lower()

def normalize_availability(product_json: Dict[str, Any]) -> Tuple[str, str]:
    """
    Heuristic normalization:
    - AVAILABLE / OUT_OF_STOCK / NOT_AVAILABLE / OTHER
    Returns (status, notes)
    """
    text = _flatten_text(product_json)

    # Not available / discontinued / not sold
    if any(kw in text for kw in [
        "not sold", "discontinued", "no longer available", "product not found", "page not found"
    ]):
        return "NOT_AVAILABLE", "Detected discontinued/not sold signals."

    # Out of stock / unavailable
    if any(kw in text for kw in [
        "out of stock", "currently unavailable", "unavailable", "sold out"
    ]):
        return "OUT_OF_STOCK", "Detected out-of-stock/unavailable signals."

    # Available / in stock / pickup/ship signals
    if any(kw in text for kw in [
        "in stock", "available", "pickup", "ship to home", "delivery"
    ]):
        return "AVAILABLE", "Detected in-stock/available/fulfillment signals."

    return "OTHER", "No clear stock signal found in response."

def pick_title(product_json: Dict[str, Any]) -> str:
    for k in ["title", "product_title", "name"]:
        v = product_json.get(k)
        if _safe_str(v):
            return _safe_str(v)
    # sometimes nested
    pdp = product_json.get("product") or {}
    if isinstance(pdp, dict):
        for k in ["title", "name"]:
            v = pdp.get(k)
            if _safe_str(v):
                return _safe_str(v)
    return ""

def pick_link(meta: Dict[str, Any], product_json: Dict[str, Any]) -> str:
    for d in [meta, product_json]:
        if isinstance(d, dict):
            for k in ["link", "url", "product_url"]:
                if _safe_str(d.get(k)):
                    return _safe_str(d.get(k))
    return ""

# -----------------------------
# App UI
# -----------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.expander("Setup (important)", expanded=True):
    st.markdown(
        """
**Do not paste your API key into chat or into the code.**  
Put it in Streamlit Secrets as `SERPAPI_API_KEY` (local: `.streamlit/secrets.toml`, cloud: App → Settings → Secrets).

This app loads your **Home Depot SKU Map** from an Excel file (Depot sheet) and lets you check availability via SerpApi.
"""
    )

colA, colB, colC = st.columns([1,1,1])
delivery_zip = colA.text_input("Delivery ZIP (optional, recommended for better availability signals)", value="")
store_id = colB.text_input("Store ID (optional)", value="")
sleep_ms = colC.slider("Throttle between requests (ms)", min_value=0, max_value=1500, value=250, step=50)

st.divider()

st.subheader("1) Load your Home Depot map (Depot sheet)")
uploaded_map = st.file_uploader("Upload SKU Map.xlsx (or keep default if running locally with the file)", type=["xlsx"])

DEFAULT_MAP_PATH = "SKU Map.xlsx"  # put next to app.py in repo/zip

def load_depot_map(file) -> pd.DataFrame:
    """
    Loads the Depot tab from the Excel map.

    Some hosting environments (notably certain Python 3.13 builds) may fail to install openpyxl.
    To keep the app running, we fall back to a bundled CSV export of the Depot tab.
    """
    expected = ["OMSID", "Internet #", "SKU #", "UPC", "GTIN", "Vendor"]

    def _validate(df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in expected if c not in df.columns]
        if missing:
            raise RuntimeError(f"Depot data missing columns: {missing}. Found columns: {list(df.columns)}")
        df = df.copy()
        for c in ["OMSID", "Internet #", "SKU #", "UPC", "GTIN"]:
            df[c] = df[c].apply(_as_int_str)
        df["Vendor"] = df["Vendor"].astype(str).str.strip()
        return df

    # If user uploaded a file, try Excel, then CSV.
    if file is not None:
        try:
            df = pd.read_excel(file, sheet_name="Depot")
            return _validate(df)
        except Exception as e:
            try:
                file.seek(0)
                df = pd.read_csv(file)
                return _validate(df)
            except Exception:
                raise RuntimeError(
                    "Could not read the uploaded map. If this environment can't install openpyxl, "
                    "upload a CSV export of the Depot sheet instead. Original error: "
                    f"{e}"
                )

    # No upload: try bundled Excel first, then bundled CSV fallback.
    try:
        df = pd.read_excel(DEFAULT_MAP_PATH, sheet_name="Depot")
        return _validate(df)
    except Exception:
        df = pd.read_csv("SKU Map - Depot.csv")
        return _validate(df)


try:
    depot_map = load_depot_map(uploaded_map)
    st.success(f"Loaded Depot map: {len(depot_map):,} rows")
    st.dataframe(depot_map.head(25), use_container_width=True, hide_index=True)
except Exception as e:
    st.error(f"Could not load Depot map. {e}")
    st.stop()

st.subheader("2) Choose what you want to check")
mode = st.radio(
    "Lookup mode",
    options=["Check items from the map (filter then run)", "Upload a list to check (CSV/XLSX)"],
    horizontal=True
)

rows_to_check = pd.DataFrame()

if mode.startswith("Check items from the map"):
    c1, c2 = st.columns([1,2])
    vendor = c1.selectbox("Vendor (optional)", options=["(All)"] + sorted(depot_map["Vendor"].dropna().unique().tolist()))
    query_text = c2.text_input("Filter (optional): match OMSID / Internet # / SKU # / UPC / GTIN", value="")
    filt = depot_map.copy()
    if vendor != "(All)":
        filt = filt[filt["Vendor"] == vendor]
    if query_text.strip():
        q = _digits_only(query_text)
        if q:
            mask = (
                filt["OMSID"].str.contains(q, na=False) |
                filt["Internet #"].str.contains(q, na=False) |
                filt["SKU #"].str.contains(q, na=False) |
                filt["UPC"].str.contains(q, na=False) |
                filt["GTIN"].str.contains(q, na=False)
            )
            filt = filt[mask]
        else:
            # non-digit query: vendor text fallback
            filt = filt[filt["Vendor"].str.contains(query_text, case=False, na=False)]
    st.caption(f"Filtered rows: {len(filt):,}")
    st.dataframe(filt.head(200), use_container_width=True, hide_index=True)
    max_n = st.number_input("Max rows to check this run", min_value=1, max_value=5000, value=min(200, max(1, len(filt))), step=1)
    rows_to_check = filt.head(int(max_n)).copy()

else:
    up = st.file_uploader("Upload list (CSV/XLSX) with any columns: OMSID, Internet #, SKU #, UPC, GTIN", type=["csv","xlsx"])
    if up is None:
        st.info("Upload a list to check, or switch to checking from the map.")
        st.stop()
    if up.name.lower().endswith(".csv"):
        df_in = pd.read_csv(up)
    else:
        df_in = pd.read_excel(up)
    # normalize column names a bit
    df_in.columns = [c.strip() for c in df_in.columns]
    # try to align to expected
    for col in ["OMSID", "Internet #", "SKU #", "UPC", "GTIN"]:
        if col in df_in.columns:
            df_in[col] = df_in[col].apply(_as_int_str)
        else:
            df_in[col] = ""
    # attach vendor from map when possible
    merged = df_in.merge(depot_map, on=["OMSID","Internet #","SKU #","UPC","GTIN"], how="left", suffixes=("","_map"))
    if "Vendor_map" in merged.columns and "Vendor" in merged.columns:
        merged["Vendor"] = merged["Vendor"].fillna(merged["Vendor_map"])
        merged.drop(columns=["Vendor_map"], inplace=True)
    rows_to_check = merged.copy()
    st.success(f"Loaded input rows: {len(rows_to_check):,}")
    st.dataframe(rows_to_check.head(200), use_container_width=True, hide_index=True)

st.subheader("3) Run availability check via SerpApi")
run = st.button("Run check", type="primary", use_container_width=True)

def build_query(row: pd.Series) -> Tuple[str, str]:
    """
    Return (query_string, query_type)
    Preference: Internet #, UPC, SKU #, GTIN, OMSID
    """
    internet = _safe_str(row.get("Internet #", ""))
    upc = _safe_str(row.get("UPC", ""))
    sku = _safe_str(row.get("SKU #", ""))
    gtin = _safe_str(row.get("GTIN", ""))
    omsid = _safe_str(row.get("OMSID", ""))

    if internet:
        return internet, "Internet #"
    if upc:
        return upc, "UPC"
    if sku:
        return sku, "SKU #"
    if gtin:
        return gtin, "GTIN"
    if omsid:
        return omsid, "OMSID"
    return "", "None"

if run:
    try:
        # force key check early
        _ = _get_api_key()
        if not _:
            st.error("Missing SERPAPI_API_KEY. Add it to Streamlit secrets and rerun.")
            st.stop()
    except Exception as e:
        st.error(str(e))
        st.stop()

    out_rows = []
    progress = st.progress(0)
    status_box = st.empty()

    total = len(rows_to_check)
    for i, (_, r) in enumerate(rows_to_check.iterrows(), start=1):
        q, qtype = build_query(r)
        base = {
            "Vendor": _safe_str(r.get("Vendor", "")),
            "OMSID": _safe_str(r.get("OMSID", "")),
            "Internet #": _safe_str(r.get("Internet #", "")),
            "SKU #": _safe_str(r.get("SKU #", "")),
            "UPC": _safe_str(r.get("UPC", "")),
            "GTIN": _safe_str(r.get("GTIN", "")),
            "QueryType": qtype,
            "Query": q,
        }

        if not q:
            out_rows.append({**base, "Status": "NOT_AVAILABLE", "Notes": "No identifier available to query.", "ProductId": "", "Title": "", "Link": ""})
            progress.progress(i/total)
            continue

        status_box.write(f"Checking {i}/{total}: {qtype}={q}")

        try:
            pid, meta = resolve_product_id_from_query(q=q, delivery_zip=delivery_zip, store_id=store_id)
            if not pid:
                out_rows.append({**base, "Status": "NOT_AVAILABLE", "Notes": "No product_id found from search results.", "ProductId": "", "Title": "", "Link": pick_link(meta, {})})
            else:
                pj = hd_product_cached(pid)
                stt, notes = normalize_availability(pj)
                out_rows.append({
                    **base,
                    "Status": stt,
                    "Notes": notes,
                    "ProductId": pid,
                    "Title": pick_title(pj),
                    "Link": pick_link(meta, pj),
                })
        except Exception as e:
            out_rows.append({**base, "Status": "OTHER", "Notes": f"Error: {e}", "ProductId": "", "Title": "", "Link": ""})

        progress.progress(i/total)
        if sleep_ms:
            time.sleep(sleep_ms/1000.0)

    result = pd.DataFrame(out_rows)
    st.success("Done.")
    st.dataframe(result, use_container_width=True, hide_index=True, height=650)

    # Summary
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(result):,}")
    c2.metric("Available", f"{(result['Status']=='AVAILABLE').sum():,}")
    c3.metric("Out of stock", f"{(result['Status']=='OUT_OF_STOCK').sum():,}")
    c4.metric("Not available", f"{(result['Status']=='NOT_AVAILABLE').sum():,}")

    csv = result.to_csv(index=False).encode("utf-8")
    st.download_button("Download results CSV", data=csv, file_name="hd_availability_results.csv", mime="text/csv", use_container_width=True)
