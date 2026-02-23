import re
import json
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

DEFAULT_ZIPS = [
    ("West", "90001"),
    ("Midwest", "60601"),
    ("Northeast", "10001"),
    ("Southeast", "30301"),
    ("Southwest", "75001"),
]

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; CornerstoneRetailMonitor/1.0)",
    "Accept-Language": "en-US,en;q=0.9",
}

REQUEST_TIMEOUT = 25

def norm_str(x):
    if pd.isna(x): return ""
    return str(x).strip()

def norm_int_str(x):
    if pd.isna(x): return ""
    s = str(x).strip()
    if s.endswith(".0"): s = s[:-2]
    return s.replace(",", "").replace(" ", "")

def norm_price(x):
    if pd.isna(x): return None
    s = str(x).replace("$","").replace(",","").strip()
    try: return float(s)
    except: return None

def sha1(s): return hashlib.sha1(s.encode()).hexdigest()

def cache_get(key, hrs):
    p = CACHE_DIR / f"{key}.json"
    if not p.exists(): return None
    if time.time() - p.stat().st_mtime > hrs*3600: return None
    return json.loads(p.read_text())

def cache_set(key,obj):
    (CACHE_DIR/f"{key}.json").write_text(json.dumps(obj))

def http_get(url, sleep, cachehrs):
    k = sha1(url)
    c = cache_get(k, cachehrs)
    if c: return c["code"], c["url"], c["txt"]
    time.sleep(sleep)
    r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    data={"code":r.status_code,"url":r.url,"txt":r.text}
    cache_set(k,data)
    return data["code"], data["url"], data["txt"]

def extract_jsonld(html):
    soup=BeautifulSoup(html,"html.parser")
    price=None; avail="Unknown"
    for s in soup.find_all("script",{"type":"application/ld+json"}):
        try:
            d=json.loads(s.text)
            nodes=d if isinstance(d,list) else [d]
            for n in nodes:
                off=n.get("offers")
                if isinstance(off,dict):
                    if "price" in off and not price:
                        price=float(str(off["price"]).replace("$",""))
                    av=off.get("availability","")
                    if "InStock" in av: avail="InStock"
                    if "OutOfStock" in av: avail="OutOfStock"
        except: pass
    return price,avail

def search_url(q):
    import requests
    return f"https://www.homedepot.com/s/{requests.utils.quote(q)}"

def find_link(html):
    soup=BeautifulSoup(html,"html.parser")
    for a in soup.find_all("a",href=True):
        if "/p/" in a["href"]:
            if a["href"].startswith("/"):
                return "https://www.homedepot.com"+a["href"]
            return a["href"]
    return None

def audit_row(row,tol,sleep,cachehrs,zips):
    sku=norm_str(row["SKU"]); inet=norm_int_str(row["Internet Number"]); upc=norm_int_str(row["UPC"])
    desired=norm_price(row["Price"])
    status="NOT_FOUND"; price=None; avail="Unknown"; url=""; matched=""
    if inet:
        matched="Internet Number"
        code,u,html=http_get(f"https://www.homedepot.com/p/{inet}",sleep,cachehrs)
        if code==200:
            price,avail=extract_jsonld(html); status="LIVE"; url=u
    if status!="LIVE":
        q=inet or upc or sku
        if q:
            matched="Search"
            code,u,html=http_get(search_url(q),sleep,cachehrs)
            link=find_link(html) if code==200 else None
            if link:
                c,u2,h2=http_get(link,sleep,cachehrs)
                if c==200:
                    price,avail=extract_jsonld(h2); status="LIVE"; url=u2
    delta=None; flag="NO_DESIRED"
    if price and desired is not None:
        delta=round(price-desired,2); flag="MATCH" if abs(delta)<=tol else "MISMATCH"
    return {
        "SKU":sku,"Internet Number":inet,"UPC":upc,"DesiredPrice":desired,
        "FoundStatus":status,"MatchedBy":matched,"Availability":avail,
        "CurrentPrice":price,"PriceDelta":delta,"PriceFlag":flag,"URL":url,
        "LastChecked":datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

st.title("Depot Free Auditor")
tol=st.number_input("Tolerance",0.0,1.0,0.01,0.01)
sleep=st.number_input("Delay",0.0,3.0,0.6,0.1)
cachehrs=st.number_input("Cache hours",1.0,48.0,8.0,1.0)
f=st.file_uploader("Upload Excel",type=["xlsx"])
if f:
    df=pd.read_excel(f)
    df=df[df["Retailer"].astype(str).str.strip()=="Depot"]
    st.write(df)
    if st.button("Run Audit"):
        res=[audit_row(r,tol,sleep,cachehrs,DEFAULT_ZIPS) for _,r in df.iterrows()]
        out=pd.DataFrame(res)
        st.dataframe(out)
        bio=out.to_excel(index=False)
