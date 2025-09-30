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
        "append_ncni": True,  # add ?NCNI-5 to reduce interstitials
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
        "search": lambda q: f"https://www.tractorsupply.com/tsc/searc
