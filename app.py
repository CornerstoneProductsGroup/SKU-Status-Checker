import streamlit as st
import pandas as pd
import requests
import re
from urllib.parse import quote_plus, urljoin
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

st.set_page_config(page_title="SKU Checker", page_icon="🛒", layout="wide")
st.title("🛒 SKU Checker Across Retailers (Sync, PDP-accurate)")
st.write("Upload a CSV/XLSX of SKUs. App searches each site, follows the first product result to its PDP, and classifies availability.")

# ---------- HTTP/session ----------
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
        total=3, backoff_factor=0.4,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD", "OPTIONS"]
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    return s

# ---------- Patterns ----------
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
        # Typical PDP anchors: /p/<slug>/<id> or full https link
        "pdp_link_pat": re.compile(r'href="(/p/[^"]+)"|href="(https?://www\.homedepot\.com/p/[^"]+)"', re.I),
        "title_clean": lambda t: re.sub(r"\s*-?\s*The Home Depot.*$", "", t, flags=re.I),
    },
    "Lowes": {
        "base": "https://www.lowes.com",
        "search": lambda q: f"https://www.lowes.com/search?searchTerm={quote_plus(q)}",
        "pdp_link_pat": re.compile(r'href="(/pd/[^"]+)"|href="(https?://www\.lowes\.com/pd/[^"]+)"', re.I),
        "title_clean": lambda t: re.sub(r"\s*at\s*Lowes\.com.*$", "", t, flags=re.I),
    },
    "TractorSupply": {
        "base": "https://www.tractorsupply.com",
        "search": lambda q: f"https://www.tractorsupply.com/tsc/search/{quote_plus(q)}",
        "pdp_link_pat": re.compile(r'href="(/tsc/product/[^"]+)"|href="(https?://www\.tractorsupply\.com/tsc/product/[^"]+)"', re.I),
        "title_clean": lambda t: re.sub(r"\s*at\s*Tractor Supply.*$", "", t, flags=re.I),
    },
}

def classify_html(html: str):
    # Quick PDP classification using text cues
    for rx in AVAIL_LIVE:
        if rx.search(html):
            return "Live / Available"
    for rx in AVAIL_NOT:
        if rx.search(html):
            return "Found but Not Available"
    return "No Results"

def find_first_pdp(search_html: str, retailer: str):
    pat = RETAILERS[retailer]["pdp_link_pat"]
    base = RETAILERS[retailer]["base"]
    m = pat.search(search_html)
    if not m:
        return None
    href = m.group(1) or m.group(2)
    return urljoin(base, href)

def clean_title(html: str, retailer: str):
    m = TITLE_PAT.search(html)
    if not m:
        return None
    raw = re.sub(r"\s+", " ", m.group(1)).strip()
    return RETAILERS[retailer]["title_clean"](raw)

def check_identifier(q: str, retailer: str, timeout: int = 20):
    s = make_session()
    try:
        # 1) Search page
        url_search = RETAILERS[retailer]["search"](q)
        r = s.get(url_search, timeout=timeout)
        search_html = r.text
        # 2) Find first PDP link and fetch PDP
        pdp = find_first_pdp(search_html, retailer)
        if not pdp:
            # Fall back to classifying the search page if no PDP link is found
            return {
                "Query": q, "Site": retailer,
                "Status": classify_html(search_html),
                "Product Name": clean_title(search_html, retailer),
                "URL": r.url, "HTTP": r.status_code,
                "Notes": "No PDP link found on search page",
            }
        r2 = s.get(pdp, timeout=timeout)
        pdp_html = r2.text
        return {
            "Query": q, "Site": retailer,
            "Status": classify_html(pdp_html),
            "Product Name": clean_title(pdp_html, retailer),
            "URL": r2.url, "HTTP": r2.status_code,
        }
    except Exception as e:
        return {
            "Query": q, "Site": retailer, "Status": "Error",
            "Product Name": None, "URL": None, "HTTP": 0, "Notes": str(e)
        }

# ---------- UI ----------
uploaded = st.file_uploader("Upload CSV/XLSX of SKUs (first column used)", type=["csv","xlsx","xls"])
use_example = st.toggle("Use example SKUs (EZC17, EZC21, EZD17, EZD21, EZL17, EZL21)")

if not uploaded and not use_example:
    st.info("Upload a file or toggle the example to proceed.")
    st.stop()

if uploaded:
    if uploaded.name.endswith(".csv"):
        df_in = pd.read_csv(uploaded)
    else:
        df_in = pd.read_excel(uploaded)
else:
    df_in = pd.DataFrame({"SKU": ["EZC17","EZC21","EZD17","EZD21","EZL17","EZL21"]})

skus = df_in.iloc[:, 0].astype(str).str.strip().replace("", pd.NA).dropna().tolist()

tabs = st.tabs(["HomeDepot.com","Lowes.com","TractorSupply.com"])
for retailer, tab in zip(["HomeDepot","Lowes","TractorSupply"], tabs):
    with tab:
        st.caption("Runs a search → follows the first product link → classifies on the product page.")
        if st.button(f"🔎 Check on {retailer}", key=f"btn_{retailer}"):
            rows = []
            prog = st.progress(0)
            for i, sku in enumerate(skus, start=1):
                rows.append(check_identifier(sku, retailer))
                prog.progress(i / max(1, len(skus)))
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            out = pd.DataFrame(rows)
            st.dataframe(out, use_container_width=True)

