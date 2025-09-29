import streamlit as st
import pandas as pd
import requests
import re
from urllib.parse import quote_plus

st.set_page_config(page_title="SKU Checker", page_icon="🛒", layout="wide")
st.title("🛒 SKU Checker Across Retailers (Sync Version)")
st.write("Upload a CSV/XLSX of SKUs. Checks HomeDepot.com, Lowes.com, TractorSupply.com.")

HEADERS = {"User-Agent": "Mozilla/5.0"}
TITLE_PAT = re.compile(r"<title>(.*?)</title>", re.S | re.I)

def classify_html(html):
    if "Add to Cart" in html or "In Stock" in html:
        return "Live / Available"
    if "Out of Stock" in html or "Unavailable" in html or "Discontinued" in html:
        return "Found but Not Available"
    return "No Results"

def check_sku(sku, site):
    if site=="HomeDepot":
        url=f"https://www.homedepot.com/s/{quote_plus(sku)}"
    elif site=="Lowes":
        url=f"https://www.lowes.com/search?searchTerm={quote_plus(sku)}"
    else:
        url=f"https://www.tractorsupply.com/tsc/search/{quote_plus(sku)}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        status = classify_html(r.text)
        return {"SKU": sku, "Site": site, "Status": status, "URL": r.url, "HTTP": r.status_code}
    except Exception as e:
        return {"SKU": sku, "Site": site, "Status": "Error", "URL": None, "HTTP": 0, "Notes": str(e)}

uploaded = st.file_uploader("Upload CSV/XLSX of SKUs", type=["csv","xlsx"])
use_example = st.toggle("Use example SKUs (EZC17, EZC21)")

if not uploaded and not use_example:
    st.stop()

if uploaded:
    if uploaded.name.endswith(".csv"):
        df_in = pd.read_csv(uploaded)
    else:
        df_in = pd.read_excel(uploaded)
else:
    df_in = pd.DataFrame({"SKU":["EZC17","EZC21","EZD17","EZD21"]})

skus = df_in.iloc[:,0].dropna().astype(str).tolist()

tabs = st.tabs(["HomeDepot.com","Lowes.com","TractorSupply.com"])
for site, tab in zip(["HomeDepot","Lowes","TractorSupply"], tabs):
    with tab:
        if st.button(f"Check on {site}", key=site):
            results=[check_sku(s, site) for s in skus]
            out=pd.DataFrame(results)
            st.dataframe(out,use_container_width=True)
