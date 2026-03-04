
import os
import time
import json
import re
from typing import Any, Dict, List, Tuple

import pandas as pd
import requests
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

APP_TITLE = "Home Depot Availability Intelligence (SerpApi)"
SERPAPI_ENDPOINT = "https://serpapi.com/search.json"

def get_secret(name: str, default: str = "") -> str:
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return str(os.getenv(name, default))

SERPAPI_KEY = get_secret("SERPAPI_API_KEY", "").strip()
DEFAULT_ZIP = get_secret("HD_DEFAULT_ZIP", "").strip()
DEFAULT_STORE = get_secret("HD_DEFAULT_STORE", "").strip()

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
    s = _safe_str(x)
    if not s:
        return ""
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    ds = _digits_only(s)
    return ds if ds else s

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename = {}
    for c in df.columns:
        c2 = c.strip()
        if c2.lower() in ["internet", "internet#", "internet number", "internet_no", "internet_no."]:
            rename[c] = "Internet #"
        if c2.lower() in ["sku", "sku#", "sku number", "skunumber", "store sku"]:
            rename[c] = "SKU #"
        if c2.lower() in ["vendor name", "vendorname"]:
            rename[c] = "Vendor"
    if rename:
        df.rename(columns=rename, inplace=True)
    for c in ["OMSID", "Internet #", "SKU #", "UPC", "GTIN", "Vendor"]:
        if c not in df.columns:
            df[c] = ""
    for c in ["OMSID", "Internet #", "SKU #", "UPC", "GTIN"]:
        df[c] = df[c].apply(_as_int_str)
    df["Vendor"] = df["Vendor"].astype(str).str.strip()
    return df

def _http_get(params: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.get(SERPAPI_ENDPOINT, params=params, timeout=45)
    if r.status_code >= 400:
        raise RuntimeError(f"SerpApi HTTP {r.status_code}: {r.text[:500]}")
    return r.json()

@st.cache_data(show_spinner=False, ttl=3600)
def hd_search_cached(q: str, delivery_zip: str = "", store_id: str = "") -> Dict[str, Any]:
    if not SERPAPI_KEY:
        raise RuntimeError("Missing SERPAPI_API_KEY in Streamlit secrets.")
    params = {"api_key": SERPAPI_KEY, "engine": "home_depot", "q": q, "country": "us"}
    if delivery_zip:
        params["delivery_zip"] = delivery_zip
    if store_id:
        params["store_id"] = store_id
    return _http_get(params)

@st.cache_data(show_spinner=False, ttl=3600)
def hd_product_cached(product_id: str) -> Dict[str, Any]:
    if not SERPAPI_KEY:
        raise RuntimeError("Missing SERPAPI_API_KEY in Streamlit secrets.")
    params = {"api_key": SERPAPI_KEY, "engine": "home_depot_product", "product_id": product_id}
    return _http_get(params)

def _find_candidates(search_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ["products", "organic_results", "shopping_results", "results"]:
        v = search_json.get(key)
        if isinstance(v, list) and v:
            return [x for x in v if isinstance(x, dict)]
    return []

def _extract_product_id(item: Dict[str, Any]) -> str:
    for k in ["product_id", "productId", "id", "item_id", "internet_number", "internetNumber"]:
        v = item.get(k)
        if _safe_str(v):
            return _as_int_str(v)
    link = _safe_str(item.get("link") or item.get("url"))
    if link:
        m = re.findall(r"(\d{7,12})", link)
        if m:
            return m[-1]
    return ""

def _score_candidate(item: Dict[str, Any], q_digits: str) -> int:
    score = 0
    try:
        text = json.dumps(item, ensure_ascii=False).lower()
    except Exception:
        text = str(item).lower()
    if q_digits and q_digits in text:
        score += 5
    title = _safe_str(item.get("title") or item.get("name"))
    if q_digits and q_digits in _digits_only(title):
        score += 10
    return score

def resolve_product_id(q: str, delivery_zip: str = "", store_id: str = "") -> Tuple[str, Dict[str, Any]]:
    sjson = hd_search_cached(q=q, delivery_zip=delivery_zip, store_id=store_id)
    items = _find_candidates(sjson)
    if not items:
        return "", {}
    qd = _digits_only(q)
    best = None
    best_score = -1
    for it in items[:20]:
        pid = _extract_product_id(it)
        if not pid:
            continue
        sc = _score_candidate(it, qd)
        if sc > best_score:
            best_score = sc
            best = it
    if best is None:
        for it in items[:20]:
            pid = _extract_product_id(it)
            if pid:
                return pid, it
        return "", items[0]
    return _extract_product_id(best), best

def _flatten(obj: Any, limit: int = 25000) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False)
    except Exception:
        s = _safe_str(obj)
    s = s.lower()
    return s[:limit] if len(s) > limit else s

def normalize_availability(product_json: Dict[str, Any]) -> Tuple[str, str]:
    txt = _flatten(product_json)
    if any(k in txt for k in ["discontinued", "no longer available", "not sold", "product not found", "page not found"]):
        return "NOT_AVAILABLE", "Detected discontinued/not-sold/not-found signals."
    if any(k in txt for k in ["out of stock", "currently unavailable", "sold out"]):
        return "OUT_OF_STOCK", "Detected out-of-stock signals."
    if any(k in txt for k in ["in stock", "available today", "pickup", "ship to home", "delivery available", "get it"]):
        return "AVAILABLE", "Detected availability/fulfillment signals."
    return "OTHER", "No clear stock signal found."

def pick_title(product_json: Dict[str, Any]) -> str:
    for k in ["title", "product_title", "name"]:
        v = product_json.get(k)
        if _safe_str(v):
            return _safe_str(v)
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

DEFAULT_MAP_XLSX = "SKU Map.xlsx"
DEFAULT_MAP_CSV = "SKU Map - Depot.csv"

def load_depot_map(uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file, sheet_name="Depot")
            return _normalize_columns(df)
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
            return _normalize_columns(df)
    try:
        df = pd.read_excel(DEFAULT_MAP_XLSX, sheet_name="Depot")
        return _normalize_columns(df)
    except Exception:
        df = pd.read_csv(DEFAULT_MAP_CSV)
        return _normalize_columns(df)

def build_query(row: pd.Series) -> Tuple[str, str]:
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

def parse_multi(text: str) -> List[str]:
    parts = []
    for p in re.split(r"[,\n;]+", text or ""):
        p = p.strip()
        if p:
            parts.append(p)
    return parts

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.header("Secrets (copy/paste)")
    st.caption("Add these to Streamlit Secrets (local `.streamlit/secrets.toml` or Cloud → Settings → Secrets).")
    st.code(
        'SERPAPI_API_KEY="PASTE_YOUR_KEY_HERE"\n'
        'HD_DEFAULT_ZIP="95355"  # optional\n'
        'HD_DEFAULT_STORE="6603" # optional\n',
        language="toml",
    )
    st.divider()
    if not SERPAPI_KEY:
        st.error("SERPAPI_API_KEY not found yet.")
    else:
        st.success("SERPAPI_API_KEY loaded.")
    max_workers = st.slider("Parallel workers", 1, 10, 5, 1)
    throttle_ms = st.slider("Throttle submit (ms)", 0, 1500, 200, 50)

st.subheader("1) Load your Home Depot map (Depot)")
uploaded_map = st.file_uploader("Upload your SKU Map.xlsx (Depot tab) OR a CSV export of Depot tab", type=["xlsx","csv"])
try:
    depot_map = load_depot_map(uploaded_map)
    st.success(f"Loaded Depot map: {len(depot_map):,} rows")
except Exception as e:
    st.error(f"Could not load Depot map. {e}")
    st.stop()

with st.expander("Preview map", expanded=False):
    st.dataframe(depot_map.head(200), use_container_width=True, hide_index=True)

st.subheader("2) Choose what to check")
mode = st.radio("Input mode", ["From map (filter then run)", "Upload list to check (CSV)"], horizontal=True)

if mode.startswith("From map"):
    c1, c2, c3 = st.columns([1,2,1])
    vendor = c1.selectbox("Vendor", ["(All)"] + sorted([v for v in depot_map["Vendor"].dropna().unique().tolist() if str(v).strip()]))
    filt_text = c2.text_input("Filter by OMSID / Internet # / SKU # / UPC / GTIN (optional)")
    max_n = c3.number_input("Max rows", 1, 5000, 200, 1)

    df = depot_map.copy()
    if vendor != "(All)":
        df = df[df["Vendor"] == vendor]
    if filt_text.strip():
        q = _digits_only(filt_text)
        if q:
            df = df[
                df["OMSID"].str.contains(q, na=False)
                | df["Internet #"].str.contains(q, na=False)
                | df["SKU #"].str.contains(q, na=False)
                | df["UPC"].str.contains(q, na=False)
                | df["GTIN"].str.contains(q, na=False)
            ]
        else:
            df = df[df["Vendor"].str.contains(filt_text, case=False, na=False)]
    st.caption(f"Filtered rows: {len(df):,}")
    st.dataframe(df.head(300), use_container_width=True, hide_index=True)
    rows = df.head(int(max_n)).copy()
else:
    up = st.file_uploader("Upload CSV with columns like OMSID / Internet # / SKU # / UPC / GTIN", type=["csv"], key="list_up")
    if up is None:
        st.info("Upload a CSV list to check.")
        st.stop()
    df_in = pd.read_csv(up)
    rows = _normalize_columns(df_in)
    st.success(f"Loaded input rows: {len(rows):,}")
    st.dataframe(rows.head(300), use_container_width=True, hide_index=True)

st.subheader("3) Location settings")
c1, c2 = st.columns(2)
zips_text = c1.text_area("ZIP codes (comma or new line)", value=DEFAULT_ZIP)
stores_text = c2.text_area("Store IDs (comma or new line)", value=DEFAULT_STORE)
zip_list = parse_multi(zips_text) or [""]
store_list = parse_multi(stores_text) or [""]

st.caption(f"Will run {len(rows):,} rows × {len(zip_list)} ZIP(s) × {len(store_list)} store(s)")

run = st.button("Run availability check", type="primary", use_container_width=True)

def one_check(row_dict: Dict[str, Any], delivery_zip: str, store_id: str) -> Dict[str, Any]:
    r = pd.Series(row_dict)
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
        "ZIP": delivery_zip,
        "StoreID": store_id,
    }
    if not q:
        return {**base, "Status": "NOT_AVAILABLE", "Notes": "No identifier.", "ProductId": "", "Title": "", "Link": ""}
    pid, meta = resolve_product_id(q=q, delivery_zip=delivery_zip, store_id=store_id)
    if not pid:
        return {**base, "Status": "NOT_AVAILABLE", "Notes": "No product_id resolved.", "ProductId": "", "Title": "", "Link": pick_link(meta, {})}
    pj = hd_product_cached(pid)
    stt, notes = normalize_availability(pj)
    return {**base, "Status": stt, "Notes": notes, "ProductId": pid, "Title": pick_title(pj), "Link": pick_link(meta, pj)}

if run:
    if not SERPAPI_KEY:
        st.error("Add SERPAPI_API_KEY to secrets first (see sidebar).")
        st.stop()

    rows_dicts = rows.to_dict(orient="records")
    total = len(rows_dicts) * len(zip_list) * len(store_list)
    progress = st.progress(0)
    status_line = st.empty()

    results = []
    futures = []
    submitted = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for zc in zip_list:
            for sid in store_list:
                for rd in rows_dicts:
                    futures.append(ex.submit(one_check, rd, zc, sid))
                    submitted += 1
                    if throttle_ms:
                        time.sleep(throttle_ms/1000.0)

        done = 0
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:
                results.append({"Status": "OTHER", "Notes": f"Error: {e}"})
            done += 1
            if total:
                progress.progress(done/total)
            if done % 20 == 0 or done == total:
                status_line.write(f"Completed {done}/{total}")

    out = pd.DataFrame(results)
    st.success("Done.")
    st.dataframe(out, use_container_width=True, hide_index=True, height=650)

    st.subheader("Dashboard")
    st.bar_chart(out["Status"].value_counts())

    st.download_button("Download CSV", data=out.to_csv(index=False).encode("utf-8"), file_name="hd_availability_results.csv", mime="text/csv", use_container_width=True)
