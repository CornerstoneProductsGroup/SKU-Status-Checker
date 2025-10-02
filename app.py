import streamlit as st
import pandas as pd
import requests
import time
import re
from urllib.parse import quote_plus
from io import BytesIO

# =========================
# App config
# =========================
st.set_page_config(page_title="SKU Checker (API-powered)", page_icon="🛒", layout="wide")
st.title("🛒 Multi-Retailer SKU Checker — API Mode (most accurate)")
st.write(
    "This version calls **SerpApi** (Home Depot) and **Apify** (Lowe’s & Tractor Supply)**, "
    "which run real browsers / structured parsers — far more accurate than raw HTML."
)

with st.sidebar:
    st.header("API Keys")
    serpapi_key = st.text_input("SerpApi key (Home Depot)", type="password")
    apify_token = st.text_input("Apify token (Lowe’s & Tractor Supply)", type="password")
    max_candidates = st.slider("PDP candidates to try (Apify)", 1, 5, 3)
    st.caption("Apify will open the search page and try the first few product links.")
    st.markdown("---")
    st.write("Upload a CSV/XLSX with SKUs in the **first column**.")

# =========================
# File upload (robust)
# =========================
uploaded = st.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx"])
use_example = st.toggle("Use example SKUs (EZC17, EZC21, EZD17, EZD21, EZL17, EZL21)")

if not uploaded and not use_example:
    st.stop()

try:
    if uploaded:
        if uploaded.name.lower().endswith(".csv"):
            df_in = pd.read_csv(uploaded)
        else:
            # needs openpyxl pinned in requirements
            import openpyxl  # noqa: F401
            df_in = pd.read_excel(uploaded, engine="openpyxl")
    else:
        df_in = pd.DataFrame({"SKU": ["EZC17","EZC21","EZD17","EZD21","EZL17","EZL21"]})
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

first_col = df_in.columns[0]
skus = (
    df_in[first_col].astype(str).str.strip().replace("", pd.NA).dropna().tolist()
)

# =========================
# Helpers
# =========================
def to_excel_bytes(df: pd.DataFrame, filename="Results") -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=filename)
    return bio.getvalue()

def norm_status(val: str | None):
    if not val:
        return None
    v = str(val).strip().lower()
    if v in ("instock", "in stock", "available", "live", "yes", "true"):
        return "Live / Available"
    if v in ("outofstock", "out of stock", "unavailable", "discontinued", "no", "false"):
        return "Found but Not Available"
    return None

# =========================
# Home Depot via SerpApi
# =========================
# Docs: https://serpapi.com/home-depot-search-api  + Product API
# Typical call: GET https://serpapi.com/search.json?engine=home_depot&q=<query>&delivery_zip=<zip>&api_key=<key>
HD_SEARCH = "https://serpapi.com/search.json"
def hd_via_serpapi(query: str, api_key: str, delivery_zip: str | None = None):
    params = {
        "engine": "home_depot",
        "q": query,
        "api_key": api_key,
    }
    if delivery_zip:
        params["delivery_zip"] = delivery_zip
    r = requests.get(HD_SEARCH, params=params, timeout=40)
    r.raise_for_status()
    data = r.json()

    # SerpApi returns 'products' for Home Depot Search API.
    # We'll take the first product that looks like a PDP.
    products = data.get("products") or []
    if not products:
        # some responses use 'organic_results'
        products = data.get("organic_results") or []

    # Pull a reasonable fieldset
    if products:
        p = products[0]
        title = p.get("title")
        link = p.get("link") or p.get("product_link") or p.get("url")
        price = p.get("price") or p.get("price_str")
        availability = p.get("availability") or p.get("availability_status")
        status = norm_status(availability)

        # If availability missing but price exists, assume Live (typical for SerpApi)
        if not status and price:
            status = "Live / Available"

        return {
            "Query": query,
            "Site": "HomeDepot",
            "Status": status or "No Results",
            "Product Name": title,
            "URL": link,
            "Price": price,
            "HTTP": 200,
            "Notes": "SerpApi Home Depot Search",
        }
    else:
        return {
            "Query": query,
            "Site": "HomeDepot",
            "Status": "No Results",
            "Product Name": None,
            "URL": None,
            "Price": None,
            "HTTP": 200,
            "Notes": "No products in SerpApi response",
        }

# =========================
# Apify (generic Web Scraper) for Lowe’s & Tractor Supply
# =========================
# We feed the retailer search URL + a pageFunction that:
#  - Clicks or selects the first few product result links
#  - Loads each PDP and extracts availability from JSON-LD/microdata/text
APIFY_ACTOR = "apify/web-scraper"
APIFY_RUN_URL = "https://api.apify.com/v2/acts/{actor}/runs"
APIFY_GET_RUN = "https://api.apify.com/v2/actor-tasks/{id}"  # not used here
APIFY_GET_ITEMS = "https://api.apify.com/v2/datasets/{dataset_id}/items?clean=true"

def apify_run_search(actor: str, token: str, start_url: str, max_candidates: int = 3):
    """
    Start an Apify web-scraper run for a retailer search URL.
    Returns (dataset_id, run_id).
    """
    # The pageFunction extracts links and then navigates to PDPs; returns a single best status.
    page_function = f"""
    async function pageFunction(context) {{
      const {{ request, log, page, $, response, crawler }} = context;
      const url = request.url;

      // helper to sleep
      const sleep = (ms) => new Promise(r => setTimeout(r, ms));

      function normAvailability(str) {{
        if (!str) return null;
        const v = String(str).toLowerCase().trim();
        if (/(instock|in stock|available|add to cart|ship to home)/.test(v)) return "Live / Available";
        if (/(outofstock|out of stock|unavailable|discontinued|temporarily unavailable)/.test(v)) return "Found but Not Available";
        return null;
      }}

      // parse JSON-LD availability
      async function availabilityFromJSONLD() {{
        try {{
          const handles = await page.$$eval('script[type="application/ld+json"]', els => els.map(e => e.textContent));
          for (const raw of handles) {{
            try {{
              const data = JSON.parse(raw);
              const arr = Array.isArray(data) ? data : [data];
              for (const obj of arr) {{
                let offers = obj.offers || obj.aggregateOffer || obj.aggregateOffers;
                if (offers) {{
                  const list = Array.isArray(offers) ? offers : [offers];
                  for (const off of list) {{
                    const av = (off && off.availability) || "";
                    const short = String(av).toLowerCase().replace("https://schema.org/","").replace("http://schema.org/","");
                    const n = normAvailability(short);
                    if (n) return n;
                  }}
                }}
              }}
            }} catch(e) {{}}
          }}
        }} catch(e) {{}}
        return null;
      }}

      async function availabilityFromMicrodata() {{
        try {{
          const href = await page.$eval('[itemprop="availability"]', el => el.getAttribute('href'));
          if (href) {{
            const short = href.toLowerCase().replace("https://schema.org/","").replace("http://schema.org/","");
            const n = normAvailability(short);
            if (n) return n;
          }}
        }} catch(e) {{}}
        return null;
      }}

      async function availabilityFromText() {{
        const html = await page.content();
        const lower = html.toLowerCase();
        if (/(add\\s*to\\s*cart|in\\s*stock|ship\\s*to\\s*home)/.test(lower)) return "Live / Available";
        if (/(out\\s*of\\s*stock|unavailable|discontinued|temporarily\\s*unavailable)/.test(lower)) return "Found but Not Available";
        return null;
      }}

      async function classifyPDP(pdpUrl) {{
        try {{
          await page.goto(pdpUrl, {{ waitUntil: 'domcontentloaded', timeout: 45000 }});
          await sleep(1500);

          let stat = await availabilityFromJSONLD();
          if (!stat) stat = await availabilityFromMicrodata();
          if (!stat) stat = await availabilityFromText();

          const title = await page.title();
          return {{ status: stat || "No Results", url: page.url(), title }};
        }} catch (e) {{
          return {{ status: "Error", url: pdpUrl, title: null, note: String(e) }};
        }}
      }}

      // If this is a search page, collect first few product links and test them.
      const isLowes = /lowes\\.com/.test(url);
      const isTsc = /tractorsupply\\.com/.test(url);

      let productLinks = [];
      if (isLowes) {{
        // Lowe's search: product anchors usually like /pd/... or have data-products
        const anchors = await page.$$eval('a[href*="/pd/"]', as => as.map(a => a.href));
        productLinks = [...new Set(anchors)].slice(0, {max_candidates});
      }} else if (isTsc) {{
        const anchors = await page.$$eval('a[href*="/tsc/product/"]', as => as.map(a => a.href));
        productLinks = [...new Set(anchors)].slice(0, {max_candidates});
      }}

      const results = [];
      for (const link of productLinks) {{
        const r = await classifyPDP(link);
        results.push(r);
        if (r.status === "Live / Available" || r.status === "Found but Not Available") break;
      }}

      if (results.length === 0) {{
        // Maybe we're already on a PDP (if user pasted PDP URL)
        const self = await classifyPDP(url);
        results.push(self);
      }}

      // push best result
      const best = results[0];
      await context.pushData(best);
    }}
    """

    payload = {
        "startUrls": [{"url": start_url}],
        "pageFunction": page_function,
        "maxConcurrency": 2,
        "initialCookies": [],
        "ignoreSslErrors": True,
        "useChrome": True,
        "proxyConfiguration": {"useApifyProxy": True},
    }

    res = requests.post(
        APIFY_RUN_URL.format(actor=actor),
        params={"token": token, "waitForFinish": 60},  # wait some seconds for fast runs
        json=payload,
        timeout=90,
    )
    res.raise_for_status()
    run = res.json().get("data", {})
    dataset_id = (run.get("defaultDatasetId") or run.get("datasetId"))
    run_id = run.get("id")
    return dataset_id, run_id

def apify_fetch_items(dataset_id: str, token: str):
    url = APIFY_GET_ITEMS.format(dataset_id=dataset_id)
    r = requests.get(url, params={"token": token}, timeout=60)
    r.raise_for_status()
    return r.json()

def lowes_via_apify(query: str, token: str, max_candidates: int = 3):
    search_url = f"https://www.lowes.com/search?searchTerm={quote_plus(query)}"
    ds, _ = apify_run_search(APIFY_ACTOR, token, search_url, max_candidates=max_candidates)
    items = apify_fetch_items(ds, token)
    if not items:
        return {
            "Query": query, "Site": "Lowes",
            "Status": "No Results", "Product Name": None,
            "URL": None, "HTTP": 200, "Notes": "Apify returned no items"
        }
    it = items[0]
    status = it.get("status")
    title = it.get("title")
    url = it.get("url")
    return {
        "Query": query, "Site": "Lowes",
        "Status": status or "No Results",
        "Product Name": title, "URL": url, "HTTP": 200,
        "Notes": f"Apify Web Scraper (candidates ≤ {max_candidates})"
    }

def tsc_via_apify(query: str, token: str, max_candidates: int = 3):
    search_url = f"https://www.tractorsupply.com/tsc/search/{quote_plus(query)}"
    ds, _ = apify_run_search(APIFY_ACTOR, token, search_url, max_candidates=max_candidates)
    items = apify_fetch_items(ds, token)
    if not items:
        return {
            "Query": query, "Site": "TractorSupply",
            "Status": "No Results", "Product Name": None,
            "URL": None, "HTTP": 200, "Notes": "Apify returned no items"
        }
    it = items[0]
    status = it.get("status")
    title = it.get("title")
    url = it.get("url")
    return {
        "Query": query, "Site": "TractorSupply",
        "Status": status or "No Results",
        "Product Name": title, "URL": url, "HTTP": 200,
        "Notes": f"Apify Web Scraper (candidates ≤ {max_candidates})"
    }

# =========================
# UI: Tabs per retailer
# =========================
tab_hd, tab_lowes, tab_tsc = st.tabs(["HomeDepot.com (SerpApi)", "Lowes.com (Apify)", "TractorSupply.com (Apify)"])

with tab_hd:
    st.caption("Uses SerpApi Home Depot API for structured results.")
    if not serpapi_key:
        st.warning("Add your **SerpApi key** in the sidebar to enable Home Depot checks.")
    else:
        if st.button("🔎 Check Home Depot"):
            rows = []
            prog = st.progress(0)
            for i, sku in enumerate(skus, start=1):
                try:
                    rows.append(hd_via_serpapi(sku, serpapi_key))
                except Exception as e:
                    rows.append({
                        "Query": sku, "Site": "HomeDepot", "Status": "Error",
                        "Product Name": None, "URL": None, "HTTP": 0, "Notes": str(e)
                    })
                prog.progress(i / max(1, len(skus)))
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            out = pd.DataFrame(rows)
            st.dataframe(out, use_container_width=True)
            st.download_button(
                "📥 Download Excel (Home Depot)",
                data=to_excel_bytes(out, "HomeDepot"),
                file_name="homedepot_status.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

with tab_lowes:
    st.caption("Uses Apify Web Scraper (headless browser) for on-page availability.")
    if not apify_token:
        st.warning("Add your **Apify token** in the sidebar to enable Lowe’s checks.")
    else:
        if st.button("🔎 Check Lowe’s"):
            rows = []
            prog = st.progress(0)
            for i, sku in enumerate(skus, start=1):
                try:
                    rows.append(lowes_via_apify(sku, apify_token, max_candidates=max_candidates))
                except Exception as e:
                    rows.append({
                        "Query": sku, "Site": "Lowes", "Status": "Error",
                        "Product Name": None, "URL": None, "HTTP": 0, "Notes": str(e)
                    })
                prog.progress(i / max(1, len(skus)))
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            out = pd.DataFrame(rows)
            st.dataframe(out, use_container_width=True)
            st.download_button(
                "📥 Download Excel (Lowe’s)",
                data=to_excel_bytes(out, "Lowes"),
                file_name="lowes_status.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

with tab_tsc:
    st.caption("Uses Apify Web Scraper (headless browser) for on-page availability.")
    if not apify_token:
        st.warning("Add your **Apify token** in the sidebar to enable Tractor Supply checks.")
    else:
        if st.button("🔎 Check Tractor Supply"):
            rows = []
            prog = st.progress(0)
            for i, sku in enumerate(skus, start=1):
                try:
                    rows.append(tsc_via_apify(sku, apify_token, max_candidates=max_candidates))
                except Exception as e:
                    rows.append({
                        "Query": sku, "Site": "TractorSupply", "Status": "Error",
                        "Product Name": None, "URL": None, "HTTP": 0, "Notes": str(e)
                    })
                prog.progress(i / max(1, len(skus)))
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            out = pd.DataFrame(rows)
            st.dataframe(out, use_container_width=True)
            st.download_button(
                "📥 Download Excel (Tractor Supply)",
                data=to_excel_bytes(out, "TractorSupply"),
                file_name="tractorsupply_status.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

st.caption("Tip: API mode is the most accurate because it uses a full browser (Apify) or structured retail API (SerpApi).")
