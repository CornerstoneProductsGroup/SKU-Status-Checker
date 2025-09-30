import streamlit as st
import pandas as pd
import requests
import re
import json
from html import unescape
from urllib.parse import quote_plus, urljoin
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# =========================
# App Config / Header
# =========================
st.set_page_config(page_title="SKU Checker", page_icon="🛒", layout="wide")
st.title("🛒 Multi-Retailer SKU Checker (Sync • PDP-accurate)")
st.write(
    "Upload a CSV/XLSX of SKUs (first column used). For each retailer, the app:\n"
    "1) runs a search, 2) tries the first few product links, 3) classifies availability **on the product page** via JSON-LD/microdata, "
    "with text fallbacks."
)

# =========================
# HTTP Session (retries)
# =========================
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "close",
}

def make_session():
    s = requests.Session()
    s.headers.update(HEADERS)
    retry = Retry(
        total=3,
        backoff_factor=0.4,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD", "OPTIONS"],
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    return s

# =========================
# Retailer Config & Regex
# =========================
TITLE_PAT = re.compile(r"<title>(.*?)</title>", re.S | re.I)

AVAIL_LIVE = [
    re.compile(r"Add\s*to\s*Cart", re.I),
    re.compile(r"Ship\s*to\s*Home", re.I),
    re.compile(r"Pickup\s*(at|in)\s*Store", re.I),
    re.compile(r"In\s*Stock", re.I),
    re.compile(r'aria-label="Add to Cart"', re.I),
]
AVAIL_NOT = [
    re.compile(r"Out\s*of\s*Stock", re.I),
    re.compile(r"Unavailable\s+at\s+this\s+time", re.I),
    re.compile(r"This\s*item\s*is\s*unavailable", re.I),
    re.compile(r"Not\s*Sold\s*in\s*Stores", re.I),
    re.compile(r"Discontinued", re.I),
    re.compile(r"Temporarily\s*Unavailable", re.I),
]

RETAILERS = {
    "HomeDepot": {
        "base": "https://www.homedepot.com",
        "search": lambda q: f"https://www.homedepot.com/s/{quote_plus(q)}?searchTerm={quote_plus(q)}",
        "pdp_link_pat": re.compile(r'href="(/p/[^"]+)"|href="(https?://www\.homedepot\.com/p/[^"]+)"', re.I),
        "title_clean": lambda t: re.sub(r"\s*-?\s*The Home Depot.*$", "", t, flags=re.I),
        "append_ncni": True,
    },
    "Lowes": {
        "base": "https://www.lowes.com",
        "search": lambda q: f"https://www.lowes.com/search?searchTerm={quote_plus(q)}",
        "pdp_link_pat": re.compile(r'href="(/pd/[^"]+)"|href="(https?://www\.lowes\.com/pd/[^"]+)"', re.I),
        "title_clean": lambda t: re.sub(r"\s*at\s*Lowes\.com.*$", "", t, flags=re.I),
        "append_ncni": False,
    },
    "TractorSupply": {
        "base": "https://www.tractorsupply.com",
        "search": lambda q: f"https://www.tractorsupply.com/tsc/search/{quote_plus(q)}",
        "pdp_link_pat": re.compile(r'href="(/tsc/product/[^"]+)"|href="(https?://www\.tractorsupply\.com/tsc/product/[^"]+)"', re.I),
        "title_clean": lambda t: re.sub(r"\s*at\s*Tractor Supply.*$", "", t, flags=re.I),
        "append_ncni": False,
    },
}

# =========================
# JSON-LD & Microdata Parsing
# =========================
AVAIL_MAP = {
    "instock": "Live / Available",
    "outofstock": "Found but Not Available",
    "discontinued": "Found but Not Available",
    "http://schema.org/instock": "Live / Available",
    "https://schema.org/instock": "Live / Available",
    "http://schema.org/outofstock": "Found but Not Available",
    "https://schema.org/outofstock": "Found but Not Available",
    "http://schema.org/discontinued": "Found but Not Available",
    "https://schema.org/discontinued": "Found but Not Available",
}

LD_JSON_PAT = re.compile(r'<script[^>]+type="application/ld\+json"[^>]*>(.*?)</script>', re.S | re.I)
MICRODATA_AVAIL_PAT = re.compile(r'itemprop=["\']availability["\'][^>]*href=["\']([^"\']+)["\']', re.I)
PRICE_PAT = re.compile(r'"price"\s*:\s*"?\$?(\d[\d\.,]*)', re.I)

def _normalize_availability(v: str | None):
    if not v:
        return None
    v = v.strip().lower().replace("http://schema.org/", "").replace("https://schema.org/", "")
    return AVAIL_MAP.get(v)

def _walk_offers(obj):
    if isinstance(obj, dict):
        if "offers" in obj:
            yield obj["offers"]
        for key in ("aggregateOffer", "aggregateOffers"):
            if key in obj:
                yield obj[key]
        for v in obj.values():
            yield from _walk_offers(v)
    elif isinstance(obj, list):
        for it in obj:
            yield from _walk_offers(it)

def classify_via_jsonld(html: str):
    for m in LD_JSON_PAT.finditer(html):
        raw = unescape(m.group(1)).strip()
        try:
            data = json.loads(raw)
        except Exception:
            continue
        for offers in _walk_offers(data):
            if isinstance(offers, list):
                for off in offers:
                    status = _normalize_availability(str(off.get("availability", "")))
                    if status:
                        return status
            elif isinstance(offers, dict):
                status = _normalize_availability(str(offers.get("availability", "")))
                if status:
                    return status
    return None

def classify_via_microdata(html: str):
    m = MICRODATA_AVAIL_PAT.search(html)
    if not m:
        return None
    return _normalize_availability(m.group(1))

def classify_html_with_fallbacks(html: str):
    via_ld = classify_via_jsonld(html)
    if via_ld:
        return via_ld
    via_micro = classify_via_microdata(html)
    if via_micro:
        return via_micro
    if PRICE_PAT.search(html):
        for rx in AVAIL_NOT:
            if rx.search(html):
                return "Found but Not Available"
        return "Live / Available"
    for rx in AVAIL_LIVE:
        if rx.search(html):
            return "Live / Available"
    for rx in AVAIL_NOT:
        if rx.search(html):
            return "Found but Not Available"
    return "No Results"

def clean_title(html: str, retailer: str):
    m = TITLE_PAT.search(html)
    if not m:
        return None
    raw = re.sub(r"\s+", " ", m.group(1)).strip()
    return RETAILERS[retailer]["title_clean"](raw)

# =========================
# PDP Links (multi-candidate)
# =========================
def find_pdp_links(search_html: str, retailer: str, max_links: int = 5):
    pat = RETAILERS[retailer]["pdp_link_pat"]
    base = RETAILERS[retailer]["base"]
    seen = set()
    out = []
    for m in pat.finditer(search_html):
        href = m.group(1) or m.group(2)
        if not href:
            continue
        url = urljoin(base, href)
        if url not in seen:
            seen.add(url)
            out.append(url)
        if len(out) >= max_links:
            break
    return out

def maybe_append_ncni(url: str, retailer: str):
    if retailer == "HomeDepot" and RETAILERS[retailer].get("append_ncni", False) and "NCNI-5" not in url:
        return f"{url}{'&' if '?' in url else '?'}NCNI-5"
    return url

def check_identifier(q: str, retailer: str, timeout: int = 20, max_candidates: int = 5):
    s = make_session()
    try:
        url_search = RETAILERS[retailer]["search"](q)
        r = s.get(url_search, timeout=timeout)
        search_html = r.text

        candidates = find_pdp_links(search_html, retailer, max_candidates)
        if not candidates:
            return {
                "Query": q, "Site": retailer,
                "Status": classify_html_with_fallbacks(search_html),
                "Product Name": clean_title(search_html, retailer),
                "URL": r.url, "HTTP": r.status_code,
                "Notes": "No PDP link found on search page",
            }

        first_resp = None
        for idx, pdp in enumerate(candidates, start=1):
            pdp_url = maybe_append_ncni(pdp, retailer)
            r2 = s.get(pdp_url, timeout=timeout)
            if first_resp is None:
                first_resp = (r2.url, r2.status_code, r2.text)
            status = classify_html_with_fallbacks(r2.text)
            if status in ("Live / Available", "Found but Not Available"):
                return {
                    "Query": q, "Site": retailer, "Status": status,
                    "Product Name": clean_title(r2.text, retailer),
                    "URL": r2.url, "HTTP": r2.status_code,
                    "Notes": f"Candidate {idx}/{len(candidates)}",
                }

        if first_resp:
            u, c, h = first_resp
            return {
                "Query": q, "Site": retailer,
                "Status": classify_html_with_fallbacks(h),
                "Product Name": clean_title(h, retailer),
                "URL": u, "HTTP": c,
                "Notes": "No definitive candidate; used first PDP",
            }

        return {
            "Query": q, "Site": retailer,
            "Status": classify_html_with_fallbacks(search_html),
            "Product Name": clean_title(search_html, retailer),
            "URL": r.url, "HTTP": r.status_code,
            "Notes": "No PDP candidates after parsing",
        }

    except Exception as e:
        return {
            "Query": q, "Site": retailer,
            "Status": "Error",
            "Product Name": None,
            "URL": None,
            "HTTP": 0,
            "Notes": str(e),
        }

# =========================
# File Upload (robust)
# =========================
with st.sidebar:
    max_candidates = st.slider("PDP candidates to try", 1, 8, 5)
    st.caption("If search shows category tiles first, trying more candidates helps.")
    st.markdown("---")

uploaded = st.file_uploader(
    "Upload CSV/XLSX of SKUs (first column used)",
    type=["csv", "xlsx", "xls"]
)
use_example = st.toggle("Use example SKUs (EZC17, EZC21, EZD17, EZD21, EZL17, EZL21)")

if not uploaded and not use_example:
    st.info("Upload a file or toggle the example to proceed.")
    st.stop()

try:
    if uploaded:
        fname = uploaded.name.lower()
        if fname.endswith(".csv"):
            df_in = pd.read_csv(uploaded)
        elif fname.endswith(".xlsx"):
            try:
                df_in = pd.read_excel(uploaded, engine="openpyxl")
            except ImportError:
                st.error("Excel support requires `openpyxl`. Add it to requirements.txt and redeploy.")
                st.stop()
        elif fname.endswith(".xls"):
            st.error("Legacy .xls files aren’t supported. Save as .xlsx or CSV instead.")
            st.stop()
        else:
            st.error("Unsupported file type. Please upload CSV or XLSX.")
            st.stop()
    else:
        df_in = pd.DataFrame({"SKU": ["EZC17","EZC21","EZD17","EZD21","EZL17","EZL21"]})
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

first_col = df_in.columns[0]
skus = (
    df_in[first_col]
    .astype(str)
    .str.strip()
    .replace("", pd.NA)
    .dropna()
    .tolist()
)

# =========================
# UI per retailer + Export
# =========================
def to_excel_bytes(df: pd.DataFrame) -> bytes:
    from io import BytesIO
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Results")
    return bio.getvalue()

tabs = st.tabs(["HomeDepot.com", "Lowes.com", "TractorSupply.com"])
for retailer, tab in zip(["HomeDepot", "Lowes", "TractorSupply"], tabs):
    with tab:
        st.caption("Search → try first few PDP links → classify availability on PDP via JSON-LD/microdata (fallback: text).")
        run = st.button(f"🔎 Check on {retailer}", key=f"btn_{retailer}")
        if run:
            rows = []
            prog = st.progress(0)
            for i, sku in enumerate(skus, start=1):
                rows.append(check_identifier(sku, retailer, max_candidates=max_candidates))
                prog.progress(i / max(1, len(skus)))
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            out = pd.DataFrame(rows)
            st.dataframe(out, use_container_width=True)

            xls = to_excel_bytes(out)
            st.download_button(
                "📥 Download Excel Results",
                data=xls,
                file_name=f"{retailer.lower()}_sku_status.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

st.caption("Tip: If you see false 'No Results', increase 'PDP candidates to try' in the sidebar.")
