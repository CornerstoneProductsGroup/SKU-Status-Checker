import re
import json
import time
import hashlib
import io
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

# -----------------------------
# Basic config (free mode)
# -----------------------------
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

HEADERS = {
    # Keep it straightforward (no stealth headers).
    "User-Agent": "Mozilla/5.0 (compatible; CornerstoneRetailMonitor/1.0)",
    "Accept-Language": "en-US,en;q=0.9",
}

REQUEST_TIMEOUT = 25

# Optional ZIP coverage columns (free mode will usually show UNKNOWN)
DEFAULT_ZIPS = [
    ("West", "90001"),
    ("Midwest", "60601"),
    ("Northeast", "10001"),
    ("Southeast", "30301"),
    ("Southwest", "75001"),
]

# -----------------------------
# Helpers
# -----------------------------
def norm_str(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()

def norm_int_str(x: Any) -> str:
    # Convert Excel numeric-looking IDs to clean strings (e.g., 1012973000.0 -> '1012973000').
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s.replace(",", "").replace(" ", "")

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def cache_get(key: str, max_age_hours: float) -> Optional[Dict[str, Any]]:
    p = CACHE_DIR / f"{key}.json"
    if not p.exists():
        return None
    age_s = time.time() - p.stat().st_mtime
    if age_s > max_age_hours * 3600:
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def cache_set(key: str, obj: Dict[str, Any]) -> None:
    (CACHE_DIR / f"{key}.json").write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")

def http_get(url: str, delay_s: float, cache_hours: float) -> Tuple[int, str, str]:
    # (status_code, final_url, html) with caching.
    key = sha1(url)
    cached = cache_get(key, max_age_hours=cache_hours)
    if cached:
        return cached["code"], cached["url"], cached["html"]

    time.sleep(max(0.0, delay_s))
    r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True)
    html = r.text if r.ok else ""
    out = {"code": r.status_code, "url": r.url, "html": html}
    cache_set(key, out)
    return out["code"], out["url"], out["html"]

# -----------------------------
# Home Depot: parsing + URLs
# -----------------------------
def search_url(query: str) -> str:
    q = requests.utils.quote(query)
    return f"https://www.homedepot.com/s/{q}"

def product_url_from_internet(internet_num: str) -> str:
    # Often resolves via redirect; if not, we fall back to search.
    return f"https://www.homedepot.com/p/{internet_num}"

def parse_jsonld_price_availability(html: str) -> Tuple[Optional[float], str]:
    # Pull price and availability from JSON-LD (best-effort).
    # Returns (price, availability) where availability is InStock/OutOfStock/Unknown.
    soup = BeautifulSoup(html, "html.parser")
    price = None
    availability = "Unknown"

    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = tag.get_text(strip=True)
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue

        nodes = data if isinstance(data, list) else [data]
        for node in nodes:
            if not isinstance(node, dict):
                continue
            offers = node.get("offers")
            if isinstance(offers, dict):
                p = offers.get("price")
                if price is None and p is not None:
                    try:
                        price = float(str(p).replace("$", "").replace(",", "").strip())
                    except Exception:
                        pass
                av = offers.get("availability", "")
                if isinstance(av, str):
                    if "InStock" in av:
                        availability = "InStock"
                    elif "OutOfStock" in av:
                        availability = "OutOfStock"
            elif isinstance(offers, list):
                for off in offers:
                    if not isinstance(off, dict):
                        continue
                    p = off.get("price")
                    if price is None and p is not None:
                        try:
                            price = float(str(p).replace("$", "").replace(",", "").strip())
                        except Exception:
                            pass
                    av = off.get("availability", "")
                    if isinstance(av, str):
                        if "InStock" in av:
                            availability = "InStock"
                        elif "OutOfStock" in av:
                            availability = "OutOfStock"

    return price, availability

def find_first_product_link_from_search(html: str) -> Optional[str]:
    # Search result pages are dynamic; when HTML contains product links,
    # pick the first /p/ link that appears to include an Internet number.
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not isinstance(href, str):
            continue
        if "/p/" in href and re.search(r"/\d{6,}(\?|$)", href):
            return "https://www.homedepot.com" + href if href.startswith("/") else href

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if isinstance(href, str) and "/p/" in href:
            return "https://www.homedepot.com" + href if href.startswith("/") else href

    return None

# -----------------------------
# Depot audit (new sheet format)
# -----------------------------
def audit_depot_row(
    row: pd.Series,
    delay_s: float,
    cache_hours: float,
    include_zip_cols: bool,
) -> Dict[str, Any]:
    omsid = norm_int_str(row.get("OMSID"))
    internet = norm_int_str(row.get("Internet #"))
    sku = norm_str(row.get("SKU #"))
    upc = norm_int_str(row.get("UPC"))
    gtin = norm_int_str(row.get("GTIN"))
    vendor = norm_str(row.get("Vendor"))

    matched_by = ""
    url = ""
    found_status = "NOT_FOUND"
    availability = "Unknown"
    price = None

    # Lookup priority: Internet # -> UPC -> GTIN -> SKU # -> OMSID (search only)
    if internet:
        matched_by = "Internet #"
        code, final_url, html = http_get(product_url_from_internet(internet), delay_s, cache_hours)
        if code == 200 and html:
            price, availability = parse_jsonld_price_availability(html)
            url = final_url
            found_status = "LIVE"
        else:
            matched_by = "Internet # (search fallback)"
            code, _, shtml = http_get(search_url(internet), delay_s, cache_hours)
            link = find_first_product_link_from_search(shtml) if (code == 200 and shtml) else None
            if link:
                c2, u2, h2 = http_get(link, delay_s, cache_hours)
                if c2 == 200 and h2:
                    price, availability = parse_jsonld_price_availability(h2)
                    url = u2
                    found_status = "LIVE"

    if found_status != "LIVE":
        for label, q in [("UPC", upc), ("GTIN", gtin), ("SKU #", sku), ("OMSID", omsid)]:
            if not q:
                continue
            matched_by = label
            code, _, shtml = http_get(search_url(q), delay_s, cache_hours)
            link = find_first_product_link_from_search(shtml) if (code == 200 and shtml) else None
            if link:
                c2, u2, h2 = http_get(link, delay_s, cache_hours)
                if c2 == 200 and h2:
                    price, availability = parse_jsonld_price_availability(h2)
                    url = u2
                    found_status = "LIVE"
                    break

    # If listing exists but price can't be read, mark for review
    if found_status == "LIVE" and price is None:
        found_status = "NEEDS_REVIEW"

    zip_cols = {}
    if include_zip_cols:
        for region, z in DEFAULT_ZIPS:
            zip_cols[f"{region}_{z}_Status"] = "UNKNOWN"

    return {
        "Retailer": "Depot",
        "Vendor": vendor,
        "OMSID": omsid,
        "Internet #": internet,
        "SKU #": sku,
        "UPC": upc,
        "GTIN": gtin,
        "FoundStatus": found_status,
        "MatchedBy": matched_by,
        "Availability": availability,
        "CurrentPrice": price,
        "ProductURL": url,
        "LastChecked": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **zip_cols,
    }

def to_excel_bytes(df: pd.DataFrame, sheet_name: str) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return bio.getvalue()

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Retailer Monitor — Depot (Free)", layout="wide")
st.title("Retailer Listing Monitor — Depot (Free / No Paid APIs)")

with st.sidebar:
    st.header("Depot Settings")
    delay_s = st.number_input("Politeness delay (sec/request)", min_value=0.0, value=0.6, step=0.1)
    cache_hours = st.number_input("Cache lifetime (hours)", min_value=0.25, value=8.0, step=0.25)
    include_zip_cols = st.checkbox("Include 5-ZIP coverage columns (free mode shows UNKNOWN)", value=True)

uploaded = st.file_uploader("Upload your SKU Map.xlsx", type=["xlsx"])
if not uploaded:
    st.stop()

try:
    df_depot = pd.read_excel(uploaded, sheet_name="Depot")
except Exception:
    df_depot = pd.read_excel(uploaded, sheet_name=0)

required = ["OMSID", "Internet #", "SKU #", "UPC", "GTIN", "Vendor"]
missing = [c for c in required if c not in df_depot.columns]
if missing:
    st.error(f"Depot sheet missing columns: {missing}")
    st.write("Found columns:", list(df_depot.columns))
    st.stop()

st.subheader("Depot Input (from Depot tab)")
st.write(f"Rows: {len(df_depot):,}")
st.dataframe(df_depot, use_container_width=True)

if st.button("Run Depot Audit", type="primary"):
    with st.spinner("Checking HomeDepot.com (free mode)…"):
        results = [
            audit_depot_row(r, float(delay_s), float(cache_hours), bool(include_zip_cols))
            for _, r in df_depot.iterrows()
        ]
        out = pd.DataFrame(results)

    st.success("Done.")
    st.subheader("Results")
    st.dataframe(out, use_container_width=True)

    exceptions = out[out["FoundStatus"].isin(["NOT_FOUND", "NEEDS_REVIEW"])].copy()
    st.subheader("Exceptions (Not Found / Needs Review)")
    st.dataframe(exceptions, use_container_width=True)

    ts = datetime.now().strftime("%Y-%m-%d")
    st.download_button(
        "Download Full Report (XLSX)",
        data=to_excel_bytes(out, "Depot_Audit"),
        file_name=f"Depot_Audit_{ts}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.download_button(
        "Download Exceptions Only (XLSX)",
        data=to_excel_bytes(exceptions, "Exceptions"),
        file_name=f"Depot_Audit_EXCEPTIONS_{ts}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.caption(
    "Free mode is best-effort: Home Depot pages can be dynamic and may not always expose price/stock in HTML. "
    "Rows marked NEEDS_REVIEW usually mean the listing was found but price couldn't be parsed."
)
