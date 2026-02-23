
import re, json, time, hashlib, io
from datetime import datetime
from pathlib import Path
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; CornerstoneRetailMonitor/1.0)",
    "Accept-Language": "en-US,en;q=0.9",
}

def norm(x): 
    if pd.isna(x): return ""
    return str(x).strip()

def norm_price(x):
    if pd.isna(x): return None
    s=str(x).replace("$","").replace(",","").strip()
    try: return float(s)
    except: return None

def sha1(s): return hashlib.sha1(s.encode()).hexdigest()

def cache_get(k):
    p=CACHE_DIR/f"{k}.json"
    if p.exists():
        return json.loads(p.read_text())
    return None

def cache_set(k,obj):
    (CACHE_DIR/f"{k}.json").write_text(json.dumps(obj))

def http_get(url, delay=0.6):
    key=sha1(url)
    c=cache_get(key)
    if c: return c["code"], c["url"], c["txt"]
    time.sleep(delay)
    r=requests.get(url,headers=HEADERS,timeout=25)
    data={"code":r.status_code,"url":r.url,"txt":r.text}
    cache_set(key,data)
    return data["code"],data["url"],data["txt"]

def extract_jsonld(html):
    soup=BeautifulSoup(html,"html.parser")
    price=None; avail="Unknown"
    for s in soup.find_all("script",{"type":"application/ld+json"}):
        try:
            data=json.loads(s.text)
            nodes=data if isinstance(data,list) else [data]
            for n in nodes:
                off=n.get("offers")
                if isinstance(off,dict):
                    if off.get("price") and price is None:
                        price=float(str(off["price"]).replace("$",""))
                    a=off.get("availability","")
                    if "InStock" in a: avail="InStock"
                    if "OutOfStock" in a: avail="OutOfStock"
        except: pass
    return price,avail

def search_url(q):
    from requests.utils import quote
    return f"https://www.homedepot.com/s/{quote(q)}"

def find_product_link(html):
    soup=BeautifulSoup(html,"html.parser")
    for a in soup.find_all("a",href=True):
        if "/p/" in a["href"]:
            if a["href"].startswith("/"):
                return "https://www.homedepot.com"+a["href"]
            return a["href"]
    return None

def audit_row(row,tol):
    sku=norm(row["SKU"]); inet=norm(row["Internet Number"]); upc=norm(row["UPC"])
    desired=norm_price(row["Price"])
    status="NOT_FOUND"; price=None; avail="Unknown"; url=""; matched=""

    if inet:
        matched="Internet Number"
        code,u,html=http_get(f"https://www.homedepot.com/p/{inet}")
        if code==200:
            price,avail=extract_jsonld(html)
            status="LIVE"; url=u

    if status!="LIVE":
        q=inet or upc or sku
        if q:
            matched="Search"
            code,u,html=http_get(search_url(q))
            link=find_product_link(html) if code==200 else None
            if link:
                c,u2,h2=http_get(link)
                if c==200:
                    price,avail=extract_jsonld(h2)
                    status="LIVE"; url=u2

    delta=None; flag="NO_DESIRED"
    if price is not None and desired is not None:
        delta=round(price-desired,2)
        flag="MATCH" if abs(delta)<=tol else "MISMATCH"

    return {
        "SKU":sku,
        "Internet Number":inet,
        "UPC":upc,
        "DesiredPrice":desired,
        "FoundStatus":status,
        "MatchedBy":matched,
        "Availability":avail,
        "CurrentPrice":price,
        "PriceDelta":delta,
        "PriceFlag":flag,
        "URL":url,
        "LastChecked":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

st.title("Depot Free Auditor")

tol=st.number_input("Price tolerance",0.0,5.0,0.01,0.01)

file=st.file_uploader("Upload Excel",type=["xlsx"])
if file:
    df=pd.read_excel(file)
    df=df[df["Retailer"].astype(str).str.strip()=="Depot"]
    st.write(df)

    if st.button("Run Audit"):
        results=[audit_row(r,tol) for _,r in df.iterrows()]
        out=pd.DataFrame(results)
        st.dataframe(out,use_container_width=True)

        buf=io.BytesIO()
        with pd.ExcelWriter(buf,engine="openpyxl") as writer:
            out.to_excel(writer,index=False)

        st.download_button(
            "Download Report",
            data=buf.getvalue(),
            file_name="Depot_Audit.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
