import os
import html
import matplotlib.pyplot as plt
import math

import pandas as pd
def avg_ignore_zeros_cols(row, cols):
    """
    Average of columns in row ignoring zeros/NaN, and ignoring the earliest week column.
    """
    use_cols = _week_cols_excluding_first(row.to_frame().T, cols)
    vals = []
    for c in use_cols:
        v = row.get(c, np.nan)
        if pd.isna(v):
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if fv == 0:
            continue
        vals.append(fv)
    return float(np.mean(vals)) if vals else 0.0


def _week_cols_excluding_first(df, week_cols):
    """
    Remove the earliest week column from week_cols (to ignore partial first week).
    Uses parsed week start date from the column name when possible.
    """
    if not week_cols:
        return week_cols
    parsed = [pd.to_datetime(c, errors="coerce") for c in week_cols]
    if all(pd.isna(p) for p in parsed):
        return week_cols[1:] if len(week_cols) > 1 else week_cols
    pairs = [(c, p) for c, p in zip(week_cols, parsed) if pd.notna(p)]
    if not pairs:
        return week_cols[1:] if len(week_cols) > 1 else week_cols
    earliest = min(pairs, key=lambda x: x[1])[0]
    return [c for c in week_cols if c != earliest]


import re
from pathlib import Path
from datetime import date, timedelta

import io
import numpy as np
import pandas as pd
import streamlit as st




def style_numeric_posneg(df: pd.DataFrame, cols: list[str]):
    def _s(v):
        try:
            if pd.isna(v):
                return ""
            x = float(v)
        except Exception:
            return ""
        if x > 0:
            return "color: #1a7f37; font-weight:700;"
        if x < 0:
            return "color: #c62828; font-weight:700;"
        return ""
    return df.style.applymap(_s, subset=[c for c in cols if c in df.columns])
# -----------------------------
# Data Coverage + Insights + One-pager Export
# -----------------------------
from datetime import datetime


def _consecutive_positive_wow(values):
    """Count consecutive week-over-week increases ending at the most recent week.
    values must be ordered chronologically (oldest -> newest).
    Non-numeric / missing values are treated as 0.0.
    """
    if values is None:
        return 0
    vals = []
    for v in values:
        try:
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                vals.append(0.0)
            else:
                vals.append(float(v))
        except Exception:
            vals.append(0.0)
    if len(vals) < 2:
        return 0
    cnt = 0
    for i in range(len(vals) - 1, 0, -1):
        if (vals[i] - vals[i - 1]) > 0:
            cnt += 1
        else:
            break
    return cnt

def build_data_coverage(df_all: pd.DataFrame) -> dict:
    if df_all is None or df_all.empty or "StartDate" not in df_all.columns:
        return {"ok": False, "msg": "No sales data loaded."}
    d = df_all.copy()
    d["StartDate"] = pd.to_datetime(d["StartDate"], errors="coerce")
    d = d[d["StartDate"].notna()].copy()
    if d.empty:
        return {"ok": False, "msg": "No valid dates in data."}
    d["Year"] = d["StartDate"].dt.year.astype(int)
    years = sorted(d["Year"].unique().tolist())
    overall = {
        "years": years,
        "min_date": d["StartDate"].min().date(),
        "max_date": d["StartDate"].max().date(),
        "rows": int(len(d)),
    }
    by_year = d.groupby("Year", as_index=False).agg(
        Weeks=("StartDate", "nunique"),
        Units=("Units", "sum"),
        Sales=("Sales", "sum"),
        MinDate=("StartDate", "min"),
        MaxDate=("StartDate", "max"),
    ).sort_values("Year")
    by_year["MinDate"] = by_year["MinDate"].dt.date.astype(str)
    by_year["MaxDate"] = by_year["MaxDate"].dt.date.astype(str)

    by_retailer = None
    if "Retailer" in d.columns:
        by_retailer = d.groupby("Retailer", as_index=False).agg(
            Weeks=("StartDate", "nunique"),
            LastWeek=("StartDate", "max"),
            Units=("Units", "sum"),
            Sales=("Sales", "sum"),
        ).sort_values("Sales", ascending=False)
        by_retailer["LastWeek"] = pd.to_datetime(by_retailer["LastWeek"]).dt.date.astype(str)

    by_vendor = None
    if "Vendor" in d.columns:
        by_vendor = d.groupby("Vendor", as_index=False).agg(
            Weeks=("StartDate", "nunique"),
            LastWeek=("StartDate", "max"),
            Units=("Units", "sum"),
            Sales=("Sales", "sum"),
        ).sort_values("Sales", ascending=False)
        by_vendor["LastWeek"] = pd.to_datetime(by_vendor["LastWeek"]).dt.date.astype(str)

    return {"ok": True, "overall": overall, "by_year": by_year, "by_retailer": by_retailer, "by_vendor": by_vendor}

def render_data_coverage_panel(df_all: pd.DataFrame):
    st.markdown("### Data coverage")
    cov = build_data_coverage(df_all)
    if not cov.get("ok"):
        st.info(cov.get("msg", "No data."))
        return
    o = cov["overall"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Years loaded", str(len(o["years"])), delta=" / ".join(map(str, o["years"])) if o["years"] else "—")
    c2.metric("Rows", fmt_int(o["rows"]))
    c3.metric("First week", str(o["min_date"]))
    c4.metric("Last week", str(o["max_date"]))

    st.markdown("#### By year")
    by_year_disp = cov["by_year"].copy()
    by_year_disp["Units"] = by_year_disp["Units"].apply(fmt_int)
    by_year_disp["Sales"] = by_year_disp["Sales"].apply(fmt_currency)
    st.dataframe(by_year_disp, use_container_width=True, hide_index=True)

    with st.expander("Coverage by retailer", expanded=False):
        if cov["by_retailer"] is None or cov["by_retailer"].empty:
            st.write("—")
        else:
            br = cov["by_retailer"].copy()
            br["Units"] = br["Units"].apply(fmt_int)
            br["Sales"] = br["Sales"].apply(fmt_currency)
            st.dataframe(br, use_container_width=True, hide_index=True, height=_table_height(br, max_px=650))

    with st.expander("Coverage by vendor", expanded=False):
        if cov["by_vendor"] is None or cov["by_vendor"].empty:
            st.write("—")
        else:
            bv = cov["by_vendor"].copy()
            bv["Units"] = bv["Units"].apply(fmt_int)
            bv["Sales"] = bv["Sales"].apply(fmt_currency)
            st.dataframe(bv, use_container_width=True, hide_index=True, height=_table_height(bv, max_px=650))

def generate_change_insights(a: pd.DataFrame, b: pd.DataFrame, label_a: str, label_b: str, value_col: str) -> list[str]:
    insights = []
    if a is None: a = pd.DataFrame()
    if b is None: b = pd.DataFrame()
    col = value_col




def render_comparison_extras(ctx: dict):
    """Top movers + Explain + One-page PDF export (Comparisons tab only)."""
    if not isinstance(ctx, dict):
        return

    a = ctx.get("a", pd.DataFrame())
    b = ctx.get("b", pd.DataFrame())
    label_a = ctx.get("label_a", "A")
    label_b = ctx.get("label_b", "B")
    value_col = ctx.get("value_col", "Sales")

    if a is None:
        a = pd.DataFrame()
    if b is None:
        b = pd.DataFrame()

    if a.empty and b.empty:
        return

    # -----------------------------
    # Top SKU movers (based on selected metric)
    # -----------------------------
    col = value_col if value_col in ("Units", "Sales") else "Sales"
    st.markdown("---")
    st.markdown("### Top SKU movers")

    if "SKU" not in a.columns and "SKU" not in b.columns:
        st.info("No SKU column available in the comparison data.")
    else:
        ga = a.groupby("SKU", as_index=False).agg(A=(col, "sum")) if (not a.empty and "SKU" in a.columns and col in a.columns) else pd.DataFrame(columns=["SKU", "A"])
        gb = b.groupby("SKU", as_index=False).agg(B=(col, "sum")) if (not b.empty and "SKU" in b.columns and col in b.columns) else pd.DataFrame(columns=["SKU", "B"])
        g = ga.merge(gb, on="SKU", how="outer").fillna(0.0)
        g["Delta"] = g["B"] - g["A"]

        up = g.sort_values("Delta", ascending=False).head(15).copy()
        down = g.sort_values("Delta", ascending=True).head(15).copy()

        fmt = fmt_currency if col == "Sales" else fmt_int
        fmt_s = fmt_currency_signed if col == "Sales" else fmt_int_signed

        def _prep(df_):
            out = df_.copy()
            out["A"] = out["A"].apply(fmt)
            out["B"] = out["B"].apply(fmt)
            out["Delta"] = out["Delta"].apply(fmt_s)
            out.rename(columns={"A": f"{label_a}", "B": f"{label_b}"}, inplace=True)
            return out[["SKU", f"{label_a}", f"{label_b}", "Delta"]]

        c1, c2 = st.columns(2)
        with c1:
            st.caption("Largest increases")
            st.dataframe(_prep(up), use_container_width=True, hide_index=True)
        with c2:
            st.caption("Largest decreases")
            st.dataframe(_prep(down), use_container_width=True, hide_index=True)

    # -----------------------------
    # Explain this change (always)
    # -----------------------------
    insights = []
    try:
        base = generate_change_insights(a, b, label_a, label_b, col)
        if isinstance(base, str):
            insights = [base] if base.strip() else []
        elif isinstance(base, list):
            insights = base
        elif base is None:
            insights = []
        else:
            try:
                insights = list(base)
            except Exception:
                insights = []
    except Exception:
        insights = []

    def _driver_lines(dim: str):
        lines = []
        if dim not in b.columns and dim not in a.columns:
            return lines

        def _dim_table(metric_col: str):
            da = a.groupby(dim, as_index=False).agg(A=(metric_col, "sum")) if (not a.empty and dim in a.columns and metric_col in a.columns) else pd.DataFrame(columns=[dim, "A"])
            db = b.groupby(dim, as_index=False).agg(B=(metric_col, "sum")) if (not b.empty and dim in b.columns and metric_col in b.columns) else pd.DataFrame(columns=[dim, "B"])
            t = da.merge(db, on=dim, how="outer").fillna(0.0)
            t["Delta"] = t["B"] - t["A"]
            up = t.sort_values("Delta", ascending=False).head(3)
            dn = t.sort_values("Delta", ascending=True).head(3)
            return up, dn

        for metric_col, fmt_signed, label in [
            ("Units", fmt_int_signed, "Units"),
            ("Sales", fmt_currency_signed, "Sales"),
        ]:
            up, dn = _dim_table(metric_col)

            up_names = ", ".join([f"{row[dim]} ({fmt_signed(row['Delta'])})" for _, row in up.iterrows() if row["Delta"] != 0])
            dn_names = ", ".join([f"{row[dim]} ({fmt_signed(row['Delta'])})" for _, row in dn.iterrows() if row["Delta"] != 0])

            if up_names:
                lines.append(f"{dim} drivers up ({label}): {up_names}.")
            if dn_names:
                lines.append(f"{dim} drivers down ({label}): {dn_names}.")

        return lines

    driver_lines = []
    driver_lines += _driver_lines("Retailer")
    driver_lines += _driver_lines("Vendor")

    with st.expander("Explain this change", expanded=True):
        for it in (insights[:10] + driver_lines)[:18]:
            st.write(f"- {it}" if it else "")
        if not insights and not driver_lines:
            st.write("—")

    # -----------------------------
    # One-page PDF
    # -----------------------------
    st.markdown("### Comparison PDF Export")
    st.caption("Exports the comparison using your current timeframe selections and any Retailer/Vendor filters.")

    if "cmp_onepager_pdf_bytes" not in st.session_state:
        st.session_state.cmp_onepager_pdf_bytes = None
        st.session_state.cmp_onepager_pdf_name = None

    build = st.button("Build Comparison One‑Pager (PDF)", use_container_width=True, key="cmp_onepager_build_btn")
    include_table = st.checkbox("Include Top changes table", value=True, key="cmp_onepager_include_table")

    if build:
        try:
            # Context
            by_dim = ctx.get("by", "Vendor")
            flt = ctx.get("filters", {}) or {}
            fR = flt.get("retailers", []) or []
            fV = flt.get("vendors", []) or []
            limit_vals = flt.get("limit_values", []) or []

            # Totals
            uA = float(a["Units"].sum()) if (a is not None and not a.empty and "Units" in a.columns) else 0.0
            uB = float(b["Units"].sum()) if (b is not None and not b.empty and "Units" in b.columns) else 0.0
            sA = float(a["Sales"].sum()) if (a is not None and not a.empty and "Sales" in a.columns) else 0.0
            sB = float(b["Sales"].sum()) if (b is not None and not b.empty and "Sales" in b.columns) else 0.0

            du = uB - uA
            ds = sB - sA
            du_pct = (du / uA) if uA else None
            ds_pct = (ds / sA) if sA else None

            def _delta_money(v, pct=None):
                s = _fmt_pdf_money(v)
                if pct is None:
                    return s
                try:
                    return f"{s} ({pct*100:+.1f}%)"
                except Exception:
                    return s

            def _delta_int(v, pct=None):
                s = _fmt_pdf_int(v)
                if pct is None:
                    return s
                try:
                    return f"{s} ({pct*100:+.1f}%)"
                except Exception:
                    return s

            kpis = [
                ("Sales A", _fmt_pdf_money(sA), None),
                ("Sales B", _fmt_pdf_money(sB), _delta_money(ds, ds_pct)),
                ("Units A", _fmt_pdf_int(uA), None),
                ("Units B", _fmt_pdf_int(uB), _delta_int(du, du_pct)),
            ]

            # Subtitle reflects filters
            parts = [f"{label_a} vs {label_b}", f"Compare by: {by_dim}"]
            if fR:
                parts.append(f"Retailer filter: {', '.join(map(str, fR[:3]))}{'…' if len(fR)>3 else ''}")
            if fV:
                parts.append(f"Vendor filter: {', '.join(map(str, fV[:3]))}{'…' if len(fV)>3 else ''}")
            if limit_vals:
                parts.append(f"Limited {by_dim}(s): {', '.join(map(str, limit_vals[:3]))}{'…' if len(limit_vals)>3 else ''}")
            parts.append(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            subtitle = " • ".join(parts)

            bullets = []
            bullets.append(f"Total sales: {_fmt_pdf_money(sA)} → {_fmt_pdf_money(sB)} ({_fmt_pdf_money(ds)}).")
            bullets.append(f"Total units: {_fmt_pdf_int(uA)} → {_fmt_pdf_int(uB)} ({_fmt_pdf_int(du)}).")

            # Build a small Top changes table
            table_df = None
            if include_table and by_dim in a.columns and by_dim in b.columns:
                ga = a.groupby(by_dim, as_index=False).agg(Units_A=("Units","sum"), Sales_A=("Sales","sum"))
                gb = b.groupby(by_dim, as_index=False).agg(Units_B=("Units","sum"), Sales_B=("Sales","sum"))
                out = ga.merge(gb, on=by_dim, how="outer").fillna(0.0)
                out["Sales_Diff"] = out["Sales_B"] - out["Sales_A"]
                out["Units_Diff"] = out["Units_B"] - out["Units_A"]
                out["_abs"] = out["Sales_Diff"].abs()
                out = out.sort_values("_abs", ascending=False).drop(columns=["_abs"]).head(10).copy()
                out["Sales_A"] = out["Sales_A"].map(_fmt_pdf_money)
                out["Sales_B"] = out["Sales_B"].map(_fmt_pdf_money)
                out["Sales_Diff"] = out["Sales_Diff"].map(_fmt_pdf_money)
                out["Units_A"] = out["Units_A"].map(_fmt_pdf_int)
                out["Units_B"] = out["Units_B"].map(_fmt_pdf_int)
                out["Units_Diff"] = out["Units_Diff"].map(_fmt_pdf_int)
                table_df = out[[by_dim, "Sales_A", "Sales_B", "Sales_Diff", "Units_A", "Units_B", "Units_Diff"]]

            
            # -----------------------------
            # Enhanced Comparison PDF (v2)
            # -----------------------------
            totals = {
                "SalesA": sA, "SalesB": sB, "SalesDiff": ds, "SalesPct": ds_pct,
                "UnitsA": uA, "UnitsB": uB, "UnitsDiff": du, "UnitsPct": du_pct,
            }

            # Retailer breakdown (ALL retailers)
            retailer_tbl = pd.DataFrame()
            try:
                if "Retailer" in a.columns or "Retailer" in b.columns:
                    ra = a.groupby("Retailer", as_index=False).agg(Units_A=("Units","sum"), Sales_A=("Sales","sum")) if (a is not None and not a.empty and "Retailer" in a.columns) else pd.DataFrame(columns=["Retailer","Units_A","Sales_A"])
                    rb = b.groupby("Retailer", as_index=False).agg(Units_B=("Units","sum"), Sales_B=("Sales","sum")) if (b is not None and not b.empty and "Retailer" in b.columns) else pd.DataFrame(columns=["Retailer","Units_B","Sales_B"])
                    retailer_tbl = ra.merge(rb, on="Retailer", how="outer").fillna(0.0)
                    retailer_tbl["Sales_Diff"] = retailer_tbl["Sales_B"] - retailer_tbl["Sales_A"]
                    retailer_tbl["Units_Diff"] = retailer_tbl["Units_B"] - retailer_tbl["Units_A"]
                    retailer_tbl = retailer_tbl.sort_values("Sales_B", ascending=False)
            except Exception:
                retailer_tbl = pd.DataFrame()

            # SKU movers (Sales delta)
            sku_tbl = pd.DataFrame()
            sku_up = pd.DataFrame()
            sku_down = pd.DataFrame()
            try:
                if "SKU" in a.columns or "SKU" in b.columns:
                    sa = a.groupby("SKU", as_index=False).agg(Units_A=("Units","sum"), Sales_A=("Sales","sum")) if (a is not None and not a.empty and "SKU" in a.columns) else pd.DataFrame(columns=["SKU","Units_A","Sales_A"])
                    sb = b.groupby("SKU", as_index=False).agg(Units_B=("Units","sum"), Sales_B=("Sales","sum")) if (b is not None and not b.empty and "SKU" in b.columns) else pd.DataFrame(columns=["SKU","Units_B","Sales_B"])
                    sku_tbl = sa.merge(sb, on="SKU", how="outer").fillna(0.0)
                    sku_tbl["Sales_Diff"] = sku_tbl["Sales_B"] - sku_tbl["Sales_A"]
                    sku_tbl["Units_Diff"] = sku_tbl["Units_B"] - sku_tbl["Units_A"]
                    sku_up = sku_tbl.sort_values("Sales_Diff", ascending=False).head(10).copy()
                    sku_down = sku_tbl.sort_values("Sales_Diff", ascending=True).head(10).copy()
            except Exception:
                sku_tbl = pd.DataFrame()
                sku_up = pd.DataFrame()
                sku_down = pd.DataFrame()

            # Momentum (within period B, respecting filters)
            momentum = pd.DataFrame()
            try:
                if b is not None and not b.empty and "SKU" in b.columns and "StartDate" in b.columns:
                    bb = b.copy()
                    bb["StartDate"] = pd.to_datetime(bb["StartDate"], errors="coerce")
                    bb = bb[bb["StartDate"].notna()].copy()
                    # Weekly sales per SKU in B
                    g = bb.groupby(["SKU","StartDate"], as_index=False).agg(Sales=("Sales","sum"))
                    rows = []
                    for sku, grp in g.groupby("SKU", sort=False):
                        grp2 = grp.sort_values("StartDate")
                        vals = grp2["Sales"].tolist()
                        streak = _consecutive_positive_wow(vals)
                        last4 = sum(vals[-4:]) if len(vals) >= 1 else 0.0
                        prev4 = sum(vals[-8:-4]) if len(vals) >= 8 else sum(vals[:-4]) if len(vals) > 4 else 0.0
                        rows.append({"SKU": sku, "Streak": int(streak), "Last4Diff": float(last4 - prev4)})
                    momentum = pd.DataFrame(rows)
                    momentum = momentum.sort_values(["Streak","Last4Diff"], ascending=[False, False])
            except Exception:
                momentum = pd.DataFrame()

            # Top 5 drivers = biggest absolute Sales changes by SKU, plus momentum hint
            drivers = pd.DataFrame()
            try:
                if sku_tbl is not None and not sku_tbl.empty:
                    drivers = sku_tbl.copy()
                    drivers["_abs"] = drivers["Sales_Diff"].abs()
                    drivers = drivers.sort_values("_abs", ascending=False).drop(columns=["_abs"]).head(5)
                    if momentum is not None and not momentum.empty:
                        drivers = drivers.merge(momentum[["SKU","Streak","Last4Diff"]], on="SKU", how="left")
                        drivers["Momentum"] = drivers.apply(lambda r: f"{int(r['Streak'])}↑, {fmt_currency_signed(r['Last4Diff'])}" if pd.notna(r.get("Streak")) else "", axis=1)
                    else:
                        drivers["Momentum"] = ""
            except Exception:
                drivers = pd.DataFrame()

            # Build PDF bytes
            pdf_bytes = make_comparison_pdf_v2(
                "Cornerstone Sales Dashboard — Comparison Report",
                subtitle,
                totals,
                retailer_tbl,
                sku_up[["SKU","Sales_Diff","Units_Diff"]] if (sku_up is not None and not sku_up.empty) else pd.DataFrame(),
                sku_down[["SKU","Sales_Diff","Units_Diff"]] if (sku_down is not None and not sku_down.empty) else pd.DataFrame(),
                drivers[["SKU","Sales_Diff","Units_Diff","Momentum"]] if (drivers is not None and not drivers.empty) else pd.DataFrame(),
                momentum,
            )
            st.session_state.cmp_onepager_pdf_bytes = pdf_bytes
            st.session_state.cmp_onepager_pdf_name = f"Comparison_{re.sub(r'[^A-Za-z0-9_\-]+','_',label_a)}_vs_{re.sub(r'[^A-Za-z0-9_\-]+','_',label_b)}.pdf"
            st.success("Comparison one‑pager is ready below.")
        except Exception as e:
            st.error(f"Comparison PDF build failed: {e}")

    if st.session_state.cmp_onepager_pdf_bytes:
        st.download_button(
            "⬇️ Download Comparison One‑Pager (PDF)",
            data=st.session_state.cmp_onepager_pdf_bytes,
            file_name=st.session_state.cmp_onepager_pdf_name or "comparison_one_pager.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="cmp_onepager_dl_btn",
        )

def make_one_pager_pdf(title: str, subtitle: str, kpis: list, bullets: list[str], table_df: pd.DataFrame|None) -> bytes:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import inch
        from reportlab.lib import colors
    except Exception:
        return b""

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    w, h = letter

    x = 0.75 * inch
    y = h - 0.75 * inch

    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, title)
    y -= 0.25 * inch
    c.setFont("Helvetica", 10)
    c.setFillColorRGB(0.2, 0.2, 0.2)
    c.drawString(x, y, subtitle)
    c.setFillColorRGB(0,0,0)
    y -= 0.35 * inch

    c.setFont("Helvetica-Bold", 9)
    box_w = (w - 1.5*inch) / 3.0
    box_h = 0.55 * inch
    cols = 3
    for i, (label, value, delta) in enumerate(kpis[:6]):
        bx = x + (i % cols) * box_w
        by = y - (i // cols) * (box_h + 0.1*inch)
        c.setStrokeColor(colors.lightgrey)
        c.rect(bx, by - box_h, box_w-6, box_h, stroke=1, fill=0)
        c.setFillColor(colors.black)
        c.drawString(bx+6, by-14, str(label)[:45])
        c.setFont("Helvetica-Bold", 12)
        c.drawString(bx+6, by-32, str(value)[:45])
        c.setFont("Helvetica", 9)
        if delta:
            c.setFillColor(colors.green if str(delta).strip().startswith("+") else colors.red if str(delta).strip().startswith("-") else colors.black)
            c.drawString(bx+6, by-46, str(delta)[:60])
            c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 9)
    y -= 2 * (box_h + 0.1*inch) + 0.1*inch

    c.setFont("Helvetica-Bold", 11)
    c.drawString(x, y, "Key insights")
    y -= 0.18*inch
    c.setFont("Helvetica", 9)
    for b in bullets[:8]:
        if y < 1.5*inch:
            break
        c.drawString(x+10, y, f"• {b}"[:110])
        y -= 0.16*inch

    if table_df is not None and not table_df.empty:
        y -= 0.10*inch
        c.setFont("Helvetica-Bold", 11)
        c.drawString(x, y, "Top rows")
        y -= 0.20*inch
        c.setFont("Helvetica", 8)
        df = table_df.head(10).copy()
        cols_show = df.columns.tolist()[:6]
        df = df[cols_show]
        col_w = (w - 1.5*inch) / len(cols_show)
        for j, cn in enumerate(cols_show):
            c.setFont("Helvetica-Bold", 8)
            c.drawString(x + j*col_w, y, str(cn)[:18])
        y -= 0.15*inch
        c.setFont("Helvetica", 8)
        for _, row in df.iterrows():
            if y < 0.9*inch:
                break
            for j, cn in enumerate(cols_show):
                c.drawString(x + j*col_w, y, str(row[cn])[:18])
            y -= 0.13*inch

    c.showPage()
    c.save()
    return buf.getvalue()




def make_comparison_pdf_v2(title: str, subtitle: str,
                           totals: dict,
                           retailer_tbl: pd.DataFrame,
                           sku_up: pd.DataFrame,
                           sku_down: pd.DataFrame,
                           drivers: pd.DataFrame,
                           momentum: pd.DataFrame) -> bytes:
    """
    Executive-style comparison PDF (matches the Weekly Summary PDF look & table styling).

    Layout:
      Page 1: Comparison Snapshot (KPI panel + Top Retailers)
      Page 2: Operational Movement (Top Drivers + Top Increases/Decreases SKUs)
      Page 3: Strategic Momentum (Momentum Leaders)
    """
    try:
        from io import BytesIO
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                        Image, KeepTogether, KeepInFrame, PageBreak)
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.pdfgen import canvas as pdfcanvas
        from datetime import datetime
        import os as _os
    except Exception:
        return b""

    def _money(v):
        try:
            v = float(v)
        except Exception:
            return "—"
        s = f"${abs(v):,.2f}"
        return f"-{s}" if v < 0 else s

    def _int(v):
        try:
            v = float(v)
        except Exception:
            return "—"
        return f"{int(round(v)):,.0f}"

    def _pct(v):
        try:
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                return "—"
            return f"{float(v)*100:+.1f}%"
        except Exception:
            return "—"

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="H1", parent=styles["Heading1"], fontSize=16, leading=18, spaceAfter=8))
    styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], fontSize=12, leading=14, spaceAfter=6))
    styles.add(ParagraphStyle(name="Body", parent=styles["Normal"], fontSize=9.5, leading=12))
    styles.add(ParagraphStyle(name="Small", parent=styles["Normal"], fontSize=8, leading=10, textColor=colors.HexColor("#6b7280")))
    styles.add(ParagraphStyle(name="Big", parent=styles["Normal"], fontSize=18, leading=20, spaceAfter=0))

    def _wow_color(v):
        """Green for positive, red for negative, neutral for zero/unknown."""
        try:
            s = str(v)
            import re as _re
            s2 = _re.sub(r"[^0-9\-\.]", "", s)
            if s2 in ("", "-", ".", "-."):
                raise ValueError("no number")
            x = float(s2)
            if x > 0:
                return colors.HexColor("#2ecc71")
            if x < 0:
                return colors.HexColor("#e74c3c")
        except Exception:
            pass
        return colors.HexColor("#111827")

    def _make_table(df: pd.DataFrame,
                    header_bg="#111827",
                    max_rows=12,
                    col_widths=None,
                    wow_cols=None,
                    right_align_cols=None):
        if df is None or df.empty:
            return Paragraph("No data.", styles["Body"])

        wow_cols = wow_cols or []
        tshow = df.copy().head(max_rows)

        # Basic formatting
        disp = tshow.copy()
        for c in disp.columns:
            cn = str(c).lower()
            ser = disp[c]
            num = pd.to_numeric(ser, errors="coerce")
            if num.notna().any():
                if ("sale" in cn) or ("revenue" in cn) or ("$" in cn):
                    disp[c] = num.map(lambda v: "—" if pd.isna(v) else (f"-${abs(v):,.2f}" if float(v) < 0 else f"${float(v):,.2f}"))
                elif ("unit" in cn) or ("qty" in cn) or ("quantity" in cn):
                    disp[c] = num.map(lambda v: "—" if pd.isna(v) else f"{int(round(float(v))):,}")
                elif ("%" in cn) or ("pct" in cn):
                    disp[c] = num.map(lambda v: "—" if pd.isna(v) else f"{float(v)*100:+.1f}%")
                else:
                    disp[c] = num.map(lambda v: "—" if pd.isna(v) else (f"{int(round(float(v))):,}" if abs(float(v)-round(float(v))) < 1e-9 else f"{float(v):,.2f}"))
            else:
                disp[c] = ser.astype(str)

        data = [disp.columns.tolist()] + disp.values.tolist()
        tbl = Table(data, hAlign="LEFT", colWidths=col_widths)

        base = [
            ("BACKGROUND",(0,0),(-1,0), colors.HexColor(header_bg)),
            ("TEXTCOLOR",(0,0),(-1,0), colors.white),
            ("FONTNAME",(0,0),(-1,0), "Helvetica-Bold"),
            ("FONTSIZE",(0,0),(-1,0), 9),
            ("GRID",(0,0),(-1,-1), 0.25, colors.HexColor("#d1d5db")),
            ("FONTNAME",(0,1),(-1,-1), "Helvetica"),
            ("FONTSIZE",(0,1),(-1,-1), 8),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, colors.HexColor("#f9fafb")]),
            ("VALIGN",(0,0),(-1,-1), "TOP"),
            ("LEFTPADDING",(0,0),(-1,-1), 4),
            ("RIGHTPADDING",(0,0),(-1,-1), 4),
            ("TOPPADDING",(0,0),(-1,-1), 3),
            ("BOTTOMPADDING",(0,0),(-1,-1), 3),
        ]

        if right_align_cols:
            for col in right_align_cols:
                if col in tshow.columns:
                    j = tshow.columns.get_loc(col)
                    base.append(("ALIGN",(j,1),(j,-1),"RIGHT"))
                    base.append(("ALIGN",(j,0),(j,0),"RIGHT"))

        # Colorize delta columns
        for col in wow_cols:
            if col in tshow.columns:
                j = tshow.columns.get_loc(col)
                for i in range(1, len(data)):
                    base.append(("TEXTCOLOR",(j,i),(j,i), _wow_color(data[i][j])))
                    base.append(("FONTNAME",(j,i),(j,i), "Helvetica-Bold"))

        tbl.setStyle(TableStyle(base))
        return tbl

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Use the same header/footer design as the weekly executive PDF
    def _on_page(canv: pdfcanvas.Canvas, doc):
        canv.saveState()
        w_, h_ = letter
        canv.setFillColor(colors.HexColor("#111827"))
        canv.rect(0, h_-0.65*inch, w_, 0.65*inch, fill=1, stroke=0)

        # Logo if present
        logo_path = "cornerstone_logo.jpg"
        if logo_path and _os.path.exists(logo_path):
            try:
                canv.drawImage(logo_path, 0.55*inch, h_-0.60*inch, width=1.3*inch, height=0.45*inch,
                               mask='auto', preserveAspectRatio=True, anchor='sw')
            except Exception:
                pass

        canv.setFillColor(colors.white)
        canv.setFont("Helvetica-Bold", 12)
        canv.drawString(2.1*inch, h_-0.40*inch, title)

        canv.setFont("Helvetica", 9)
        canv.drawRightString(w_-0.55*inch, h_-0.40*inch, f"Generated: {generated}")

        canv.setFillColor(colors.HexColor("#6b7280"))
        canv.setFont("Helvetica", 8)
        canv.drawString(0.55*inch, 0.45*inch, "Cornerstone Products Group – Confidential (Internal Use Only)")
        canv.drawRightString(w_-0.55*inch, 0.45*inch, f"Page {doc.page}")
        canv.restoreState()

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
                            leftMargin=0.55*inch, rightMargin=0.55*inch,
                            topMargin=0.85*inch, bottomMargin=0.7*inch)
    story = []

    # -------------------------
    # PAGE 1 — Comparison Snapshot
    # -------------------------
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("Comparison Snapshot", styles["H1"]))
    if subtitle:
        story.append(Paragraph(subtitle, styles["Small"]))
        story.append(Spacer(1, 0.10*inch))

    # KPI / Totals panel (styled like Executive Snapshot)
    salesA = totals.get("SalesA", 0)
    salesB = totals.get("SalesB", 0)
    salesD = totals.get("SalesDiff", 0)
    salesP = totals.get("SalesPct", None)

    unitsA = totals.get("UnitsA", 0)
    unitsB = totals.get("UnitsB", 0)
    unitsD = totals.get("UnitsDiff", 0)
    unitsP = totals.get("UnitsPct", None)

    # Two "hero" KPIs (Sales, Units) displayed bigger
    hero_tbl = Table([
        ["Sales", Paragraph(f"<b>{_money(salesB)}</b>", styles["Big"]), Paragraph(f"<font color='{_wow_color(salesD).hexval()}'><b>{_money(salesD)}</b></font>  {_pct(salesP)}", styles["Body"])],
        ["Units", Paragraph(f"<b>{_int(unitsB)}</b>", styles["Big"]), Paragraph(f"<font color='{_wow_color(unitsD).hexval()}'><b>{_int(unitsD)}</b></font>  {_pct(unitsP)}", styles["Body"])],
    ], colWidths=[0.9*inch, 2.2*inch, 2.6*inch])
    hero_tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1), colors.HexColor("#f3f4f6")),
        ("BOX",(0,0),(-1,-1), 0.5, colors.HexColor("#d1d5db")),
        ("INNERGRID",(0,0),(-1,-1), 0.25, colors.HexColor("#e5e7eb")),
        ("FONTNAME",(0,0),(0,-1), "Helvetica-Bold"),
        ("VALIGN",(0,0),(-1,-1), "MIDDLE"),
        ("LEFTPADDING",(0,0),(-1,-1), 8),
        ("RIGHTPADDING",(0,0),(-1,-1), 8),
        ("TOPPADDING",(0,0),(-1,-1), 8),
        ("BOTTOMPADDING",(0,0),(-1,-1), 8),
    ]))
    story.append(hero_tbl)
    story.append(Spacer(1, 0.18*inch))

    # Small details (A vs B) like the executive snapshot style
    kpi_lines = [
        ["Sales A", _money(salesA)],
        ["Sales B", _money(salesB)],
        ["Units A", _int(unitsA)],
        ["Units B", _int(unitsB)],
    ]
    kpi_tbl = Table(kpi_lines, colWidths=[1.25*inch, 1.55*inch])
    kpi_tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1), colors.HexColor("#ffffff")),
        ("BOX",(0,0),(-1,-1), 0.5, colors.HexColor("#d1d5db")),
        ("INNERGRID",(0,0),(-1,-1), 0.25, colors.HexColor("#e5e7eb")),
        ("FONTNAME",(0,0),(0,-1), "Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1), 9),
        ("LEFTPADDING",(0,0),(-1,-1), 6),
        ("RIGHTPADDING",(0,0),(-1,-1), 6),
        ("TOPPADDING",(0,0),(-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
    ]))

    story.append(kpi_tbl)
    story.append(Spacer(1, 0.18*inch))

    # Top Retailers (ALL retailers, styled like the executive tables)
    story.append(Paragraph("Retailers (A vs B)", styles["H2"]))
    rt = retailer_tbl.copy() if retailer_tbl is not None else pd.DataFrame()
    if not rt.empty:
        # Ensure consistent column names
        # Expecting: Retailer, UnitsA, SalesA, UnitsB, SalesB, Sales_Diff, Units_Diff, Sales_Pct
        # We'll try to map common variants.
        colmap = {c.lower(): c for c in rt.columns}
        def _pick(*names):
            for nm in names:
                if nm.lower() in colmap:
                    return colmap[nm.lower()]
            return None

        c_ret = _pick("Retailer", "Store")
        c_ua  = _pick("UnitsA", "Units_A", "Units (A)", "Units A")
        c_sa  = _pick("SalesA", "Sales_A", "Sales (A)", "Sales A")
        c_ub  = _pick("UnitsB", "Units_B", "Units (B)", "Units B")
        c_sb  = _pick("SalesB", "Sales_B", "Sales (B)", "Sales B")
        c_sd  = _pick("Sales_Diff", "SalesDiff", "Sales Δ", "DeltaSales", "Sales Change")
        c_ud  = _pick("Units_Diff", "UnitsDiff", "Units Δ", "DeltaUnits", "Units Change")
        c_sp  = _pick("SalesPct", "Sales_Pct", "%∆Sales", "PctSales", "Sales %")

        use_cols = [c_ret, c_ub, c_sb, c_ud, c_sd, c_sp]
        use_cols = [c for c in use_cols if c is not None]

        rt2 = rt[use_cols].copy()
        # Rename columns for display
        rename = {}
        if c_ret: rename[c_ret] = "Retailer"
        if c_ub: rename[c_ub] = "Units"
        if c_sb: rename[c_sb] = "Sales"
        if c_ud: rename[c_ud] = "WoW Units"
        if c_sd: rename[c_sd] = "WoW $ Diff"
        if c_sp: rename[c_sp] = "%∆Sales"
        rt2 = rt2.rename(columns=rename)

        story.append(_make_table(rt2, max_rows=60,
                                 wow_cols=[c for c in ["WoW $ Diff", "WoW Units", "%∆Sales"] if c in rt2.columns],
                                 right_align_cols=[c for c in rt2.columns if c != "Retailer"]))
    else:
        story.append(Paragraph("No retailer rows for this selection.", styles["Body"]))

    story.append(PageBreak())

    # -------------------------
    # PAGE 2 — Operational Movement
    # -------------------------
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("Operational Movement", styles["H1"]))

    # Top Drivers
    story.append(Paragraph("Top 5 Drivers of Sales Change", styles["H2"]))
    ddf = drivers.copy() if drivers is not None else pd.DataFrame()
    if not ddf.empty:
        # Normalize expected columns
        d_cols = []
        for cand in ["SKU", "Sales_Diff", "Units_Diff", "Momentum"]:
            if cand in ddf.columns:
                d_cols.append(cand)
        dshow = ddf[d_cols].copy()
        dshow = dshow.rename(columns={"Sales_Diff":"WoW Sales ($)", "Units_Diff":"WoW Units"})
        story.append(_make_table(dshow, max_rows=5,
                                 wow_cols=[c for c in ["WoW Sales ($)", "WoW Units"] if c in dshow.columns],
                                 right_align_cols=[c for c in dshow.columns if c != "SKU" and c != "Momentum"]))
    else:
        story.append(Paragraph("No driver rows for this selection.", styles["Body"]))

    story.append(Spacer(1, 0.15*inch))

    # SKU movers (up/down)
    story.append(Paragraph("Top Increase SKUs", styles["H2"]))
    up = sku_up.copy() if sku_up is not None else pd.DataFrame()
    if not up.empty:
        cols = [c for c in ["SKU","Sales_Diff","Units_Diff"] if c in up.columns]
        up2 = up[cols].copy().rename(columns={"Sales_Diff":"WoW Sales ($)","Units_Diff":"WoW Units"})
        story.append(_make_table(up2, max_rows=10,
                                 wow_cols=[c for c in ["WoW Sales ($)","WoW Units"] if c in up2.columns],
                                 right_align_cols=[c for c in up2.columns if c != "SKU"]))
    else:
        story.append(Paragraph("—", styles["Body"]))

    story.append(Spacer(1, 0.12*inch))

    story.append(Paragraph("Top Decrease SKUs", styles["H2"]))
    dn = sku_down.copy() if sku_down is not None else pd.DataFrame()
    if not dn.empty:
        cols = [c for c in ["SKU","Sales_Diff","Units_Diff"] if c in dn.columns]
        dn2 = dn[cols].copy().rename(columns={"Sales_Diff":"WoW Sales ($)","Units_Diff":"WoW Units"})
        story.append(_make_table(dn2, max_rows=10,
                                 wow_cols=[c for c in ["WoW Sales ($)","WoW Units"] if c in dn2.columns],
                                 right_align_cols=[c for c in dn2.columns if c != "SKU"]))
    else:
        story.append(Paragraph("—", styles["Body"]))

    story.append(PageBreak())

    # -------------------------
    # PAGE 3 — Strategic Momentum
    # -------------------------
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("Strategic Momentum", styles["H1"]))
    story.append(Paragraph("Momentum Leaders (Selected Filter)", styles["H2"]))

    mm = momentum.copy() if momentum is not None else pd.DataFrame()
    if not mm.empty:
        # Prefer the same columns as the executive momentum table if present
        pref = [c for c in ["SKU","Momentum","Up Weeks","Down Weeks","Units_Last","Sales_Last","Streak"] if c in mm.columns]
        if not pref:
            pref = mm.columns.tolist()
        mshow = mm[pref].copy().head(15)

        # Rename common variants
        ren = {}
        if "Streak" in mshow.columns: ren["Streak"] = "Momentum"
        if "Sales_Last" in mshow.columns: ren["Sales_Last"] = "Sales_Last"
        if "Units_Last" in mshow.columns: ren["Units_Last"] = "Units_Last"
        mshow = mshow.rename(columns=ren)

        story.append(_make_table(mshow, max_rows=15,
                                 right_align_cols=[c for c in mshow.columns if c != "SKU"]))
    else:
        story.append(Paragraph("No momentum rows for this selection.", styles["Body"]))

    try:
        doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    except Exception:
        return b""

    buf.seek(0)
    return buf.getvalue()


def render_comparison_retailer_vendor():
        st.subheader("Comparison")
        st.session_state["cmp_ctx"] = {}

        if df.empty:
            st.info("No sales data yet.")
            return

        d = df_all.copy()
        d["StartDate"] = pd.to_datetime(d["StartDate"], errors="coerce")
        d = d[d["StartDate"].notna()].copy()

        # Build month + year options across ALL years in the store
        d["MonthP"] = d["StartDate"].dt.to_period("M")
        months = sorted(d["MonthP"].unique().tolist())
        month_labels = [m.to_timestamp().strftime("%B %Y") for m in months]
        label_to_period = dict(zip(month_labels, months))

        d["Year"] = d["StartDate"].dt.year.astype(int)
        years = sorted(d["Year"].dropna().unique().tolist())


        # Optional filters (apply to BOTH A and B)
        with st.expander("Filters (optional)", expanded=False):
            r_opts = sorted([str(x) for x in d.get("Retailer", pd.Series([], dtype="object")).dropna().unique().tolist() if str(x).strip()])
            v_opts = sorted([str(x) for x in d.get("Vendor", pd.Series([], dtype="object")).dropna().unique().tolist() if str(x).strip()])
            f_retailers = st.multiselect("Filter Retailer(s)", options=r_opts, default=st.session_state.get("cmp_filter_retailers", []), key="cmp_filter_retailers")
            f_vendors = st.multiselect("Filter Vendor(s)", options=v_opts, default=st.session_state.get("cmp_filter_vendors", []), key="cmp_filter_vendors")

        if f_retailers:
            d = d[d["Retailer"].astype(str).isin([str(x) for x in f_retailers])].copy()
        if f_vendors:
            d = d[d["Vendor"].astype(str).isin([str(x) for x in f_vendors])].copy()


        mode = st.radio(
            "Compare type",
            options=["A vs B (Months)", "A vs B (Years)", "Multi-year (high/low highlight)", "Multi-month across years"],
            index=0,
            horizontal=True,
            key="cmp_mode_v2"
        )

        c1, c2, c3 = st.columns([2, 2, 1])
        with c3:
            by = st.selectbox("Compare by", ["Retailer", "Vendor"], key="cmp_by_v2")

        # Optional limiter list
        if by == "Retailer":
            options = sorted(d["Retailer"].dropna().unique().tolist())
        else:
            options = sorted([v for v in d["Vendor"].dropna().unique().tolist() if str(v).strip()])

        sel = st.multiselect(f"Limit to {by}(s) (optional)", options=options, key="cmp_limit_v2")

        # -------------------------
        # Helper: render A vs B table
        # -------------------------
        def _render_a_vs_b(da: pd.DataFrame, db: pd.DataFrame, label_a: str, label_b: str):
            ga = da.groupby(by, as_index=False).agg(Units_A=("Units","sum"), Sales_A=("Sales","sum"))
            gb = db.groupby(by, as_index=False).agg(Units_B=("Units","sum"), Sales_B=("Sales","sum"))

            out = ga.merge(gb, on=by, how="outer").fillna(0.0)
            out["Units_Diff"] = out["Units_A"] - out["Units_B"]
            out["Sales_Diff"] = out["Sales_A"] - out["Sales_B"]
            out["Units_%"] = out["Units_Diff"] / out["Units_B"].replace(0, np.nan)
            out["Sales_%"] = out["Sales_Diff"] / out["Sales_B"].replace(0, np.nan)

            total = {
                by: "TOTAL",
                "Units_A": out["Units_A"].sum(),
                "Sales_A": out["Sales_A"].sum(),
                "Units_B": out["Units_B"].sum(),
                "Sales_B": out["Sales_B"].sum(),
            }
            total["Units_Diff"] = total["Units_A"] - total["Units_B"]
            total["Sales_Diff"] = total["Sales_A"] - total["Sales_B"]
            total["Units_%"] = total["Units_Diff"] / total["Units_B"] if total["Units_B"] else np.nan
            total["Sales_%"] = total["Sales_Diff"] / total["Sales_B"] if total["Sales_B"] else np.nan

            out = pd.concat([out, pd.DataFrame([total])], ignore_index=True)

            disp = out[[by,"Units_A","Sales_A","Units_B","Sales_B","Units_Diff","Units_%","Sales_Diff","Sales_%"]].copy()
            disp = disp.rename(columns={
                "Units_A": f"Units ({label_a})",
                "Sales_A": f"Sales ({label_a})",
                "Units_B": f"Units ({label_b})",
                "Sales_B": f"Sales ({label_b})",
            })

            sty = disp.style.format({
                f"Units ({label_a})": fmt_int,
                f"Units ({label_b})": fmt_int,
                "Units_Diff": fmt_int,
                "Units_%": lambda v: f"{v*100:.1f}%" if pd.notna(v) else "—",
                f"Sales ({label_a})": fmt_currency,
                f"Sales ({label_b})": fmt_currency,
                "Sales_Diff": fmt_currency,
                "Sales_%": lambda v: f"{v*100:.1f}%" if pd.notna(v) else "—",
            }).applymap(lambda v: f"color: {_color(v)};", subset=["Units_Diff","Sales_Diff"])

            st.dataframe(sty, use_container_width=True, hide_index=True)

        # -------------------------
        # Mode 1: A vs B (Months)
        # -------------------------
        if mode == "A vs B (Months)":
            with c1:
                a_pick = st.multiselect(
                    "Selection A (one or more months)",
                    options=month_labels,
                    default=month_labels[-1:] if month_labels else [],
                    key="cmp_a_months_v2"
                )
            with c2:
                b_pick = st.multiselect(
                    "Selection B (one or more months)",
                    options=month_labels,
                    default=month_labels[-2:-1] if len(month_labels) >= 2 else [],
                    key="cmp_b_months_v2"
                )

            a_periods = [label_to_period[x] for x in a_pick if x in label_to_period]
            b_periods = [label_to_period[x] for x in b_pick if x in label_to_period]

            # Store for Top SKU movers table
            st.session_state["movers_a_periods"] = [str(p) for p in a_periods]
            st.session_state["movers_b_periods"] = [str(p) for p in b_periods]

            if not a_periods or not b_periods:
                st.info("Pick at least one month in Selection A and Selection B.")
                return

            da = d[d["MonthP"].isin(a_periods)]
            db = d[d["MonthP"].isin(b_periods)]

            if sel:
                da = da[da[by].isin(sel)]
                db = db[db[by].isin(sel)]

            label_a = " + ".join(a_pick) if a_pick else "A"
            label_b = " + ".join(b_pick) if b_pick else "B"
            _render_a_vs_b(da, db, label_a, label_b)
            st.session_state["cmp_ctx"] = {"a": da.copy(), "b": db.copy(), "label_a": label_a, "label_b": label_b, "value_col": "Sales", "by": by, "filters": {"retailers": f_retailers, "vendors": f_vendors, "limit_by": by, "limit_values": sel}}
            return

        # -------------------------
        # Mode 2: A vs B (Years)
        # Example: compare (2023+2024) vs (2024+2025)
        # -------------------------
        if mode == "A vs B (Years)":
            with c1:
                years_a = st.multiselect(
                    "Selection A (one or more years)",
                    options=years,
                    default=years[-2:-1] if len(years) >= 2 else years,
                    key="cmp_years_a_v2"
                )
            with c2:
                years_b = st.multiselect(
                    "Selection B (one or more years)",
                    options=years,
                    default=years[-1:] if years else [],
                    key="cmp_years_b_v2"
                )

            if not years_a or not years_b:
                st.info("Pick at least one year in Selection A and Selection B.")
                return

            da = d[d["Year"].isin([int(y) for y in years_a])]
            db = d[d["Year"].isin([int(y) for y in years_b])]

            # Store for Top SKU movers table (months within those years)
            try:
                st.session_state["movers_a_periods"] = [str(p) for p in sorted(da["MonthP"].unique().tolist())] if "MonthP" in da.columns else []
                st.session_state["movers_b_periods"] = [str(p) for p in sorted(db["MonthP"].unique().tolist())] if "MonthP" in db.columns else []
            except Exception:
                st.session_state["movers_a_periods"] = []
                st.session_state["movers_b_periods"] = []

            if sel:
                da = da[da[by].isin(sel)]
                db = db[db[by].isin(sel)]

            label_a = " + ".join([str(y) for y in years_a])
            label_b = " + ".join([str(y) for y in years_b])
            _render_a_vs_b(da, db, label_a, label_b)
            st.session_state["cmp_ctx"] = {"a": da.copy(), "b": db.copy(), "label_a": label_a, "label_b": label_b, "value_col": "Sales", "by": by, "filters": {"retailers": f_retailers, "vendors": f_vendors, "limit_by": by, "limit_values": sel}}
            return


        # -------------------------
        # Mode 2b: Multi-month across years (pick Month+Year periods)
        # -------------------------
        if mode == "Multi-month across years":
            month_names = ["January","February","March","April","May","June","July","August","September","October","November","December"]
            period_options = []
            for y in years:
                for mname in month_names:
                    period_options.append(f"{mname} {y}")

            with c1:
                periods_pick = st.multiselect(
                    "Month + Year periods",
                    options=period_options,
                    default=[period_options[-1]] if period_options else [],
                    key="cmp_mm_periods",
                )
            with c2:
                metric = st.selectbox(
                    "Highlight based on",
                    options=["Sales", "Units"],
                    index=0,
                    key="cmp_mm_metric",
                )
            with c3:
                topn = st.selectbox("Show", options=[25, 50, 100, 250], index=1, key="cmp_mm_topn")

            if not periods_pick:
                st.info("Pick at least one Month + Year period.")
                return

            month_to_num = {m:i+1 for i,m in enumerate(month_names)}
            pairs = []
            for p in periods_pick:
                try:
                    parts = p.split(" ")
                    mname = " ".join(parts[:-1])
                    yy = int(parts[-1])
                    mn = month_to_num.get(mname, None)
                    if mn is not None:
                        pairs.append((yy, mn, p))
                except Exception:
                    continue
            if not pairs:
                st.info("No valid periods selected.")
                return

            dd = d.copy()
            mask = False
            for (yy, mn, _) in pairs:
                mask = mask | ((dd["Year"] == int(yy)) & (dd["StartDate"].dt.month == int(mn)))
            dd = dd[mask].copy()
            if sel:
                dd = dd[dd[by].isin(sel)]

            if dd.empty:
                st.info("No data found for those periods with current filters.")
                return

            pieces = []
            for (yy, mn, lab_full) in pairs:
                lab = lab_full.replace("January","Jan").replace("February","Feb").replace("March","Mar").replace("April","Apr").replace("June","Jun").replace("July","Jul").replace("August","Aug").replace("September","Sep").replace("October","Oct").replace("November","Nov").replace("December","Dec")
                dyy = dd[(dd["Year"] == int(yy)) & (dd["StartDate"].dt.month == int(mn))].copy()
                gy = dyy.groupby(by, as_index=False).agg(**{
                    f"Units_{lab}": ("Units", "sum"),
                    f"Sales_{lab}": ("Sales", "sum"),
                })
                pieces.append((lab, gy))

            out = pieces[0][1]
            for lab, p in pieces[1:]:
                out = out.merge(p, on=by, how="outer")
            out = out.fillna(0.0)

            total = {by: "TOTAL"}
            for c in out.columns:
                if c == by:
                    continue
                total[c] = float(out[c].sum()) if pd.api.types.is_numeric_dtype(out[c]) else ""
            out = pd.concat([out, pd.DataFrame([total])], ignore_index=True)

            cols = [by]
            for lab, _ in pieces:
                cols += [f"Units_{lab}", f"Sales_{lab}"]
            disp = out[cols].copy()

            metric_cols = [c for c in disp.columns if c.startswith(metric + "_")]
            if metric_cols:
                disp = disp.sort_values(metric_cols[-1], ascending=False).head(int(topn)).copy()

            spark_chars = ["▁","▂","▃","▄","▅","▆","▇","█"]
            def _spark(vals):
                vals = [float(v) if v is not None and not pd.isna(v) else np.nan for v in vals]
                if len(vals) == 0 or all(pd.isna(v) for v in vals):
                    return ""
                vmin = np.nanmin(vals); vmax = np.nanmax(vals)
                if np.isnan(vmin) or np.isnan(vmax) or np.isclose(vmin, vmax):
                    return "▁" * len(vals)
                out_s=[]
                for v in vals:
                    if pd.isna(v):
                        out_s.append(" "); continue
                    t=(v-vmin)/(vmax-vmin)
                    idx=int(round(t*(len(spark_chars)-1)))
                    idx=max(0,min(len(spark_chars)-1,idx))
                    out_s.append(spark_chars[idx])
                return "".join(out_s)

            def _cagr(a,b,periods):
                try:
                    a=float(a); b=float(b)
                except Exception:
                    return np.nan
                if a <= 0 or b <= 0 or periods <= 0:
                    return np.nan
                return (b/a)**(1.0/periods)-1.0

            if metric_cols:
                series_vals = disp[metric_cols].copy()
                disp["Spark"] = series_vals.apply(lambda r: _spark(r.tolist()), axis=1)
                periods_n = max(1, len(metric_cols)-1)
                disp["CAGR %"] = series_vals.apply(lambda r: _cagr(r[metric_cols[0]], r[metric_cols[-1]], periods_n), axis=1)

            def _hl(row):
                styles=[""]*len(row)
                if not metric_cols:
                    return styles
                vals=[]
                for c in metric_cols:
                    try:
                        vals.append(float(row[c]))
                    except Exception:
                        vals.append(np.nan)
                if len(vals)==0 or all(pd.isna(v) for v in vals):
                    return styles
                vmax=np.nanmax(vals); vmin=np.nanmin(vals)
                for i,c in enumerate(row.index):
                    if c in metric_cols:
                        v=float(row[c]) if pd.notna(row[c]) else np.nan
                        if pd.notna(v) and np.isclose(v,vmax):
                            styles[i]="background-color: rgba(0, 200, 0, 0.18); font-weight: 600;"
                        elif pd.notna(v) and np.isclose(v,vmin):
                            styles[i]="background-color: rgba(220, 0, 0, 0.14);"
                return styles

            fmt_map = {c: fmt_int for c in disp.columns if c.startswith("Units_")}
            fmt_map.update({c: fmt_currency for c in disp.columns if c.startswith("Sales_")})
            if "CAGR %" in disp.columns:
                fmt_map["CAGR %"] = lambda v: f"{v*100:.1f}%" if pd.notna(v) else "—"

            st.caption("Multi-month across years (selected Month+Year periods)")
            sty = disp.style.format(fmt_map)
            if metric_cols:
                sty = sty.apply(_hl, axis=1)
            st.dataframe(sty, use_container_width=True, hide_index=True)

            first = pairs[0]; last = pairs[-1]
            a_ctx = dd[(dd["Year"] == int(first[0])) & (dd["StartDate"].dt.month == int(first[1]))].copy()
            b_ctx = dd[(dd["Year"] == int(last[0])) & (dd["StartDate"].dt.month == int(last[1]))].copy()
            st.session_state["cmp_ctx"] = {"a": a_ctx, "b": b_ctx, "label_a": first[2], "label_b": last[2], "value_col": metric, "by": by, "filters": {"retailers": f_retailers, "vendors": f_vendors, "limit_by": by, "limit_values": sel}}
            return

        # -------------------------
        # Mode 3: Multi-year highlight table
        # - pick 2..5 years
        # - show Units_YYYY and Sales_YYYY columns
        # - highlight highest and lowest per row (for Sales columns)
        # -------------------------
        with c1:
            years_pick = st.multiselect(
                "Years to view (pick 2 to 5)",
                options=years,
                default=years[-3:] if len(years) >= 3 else years,
                key="cmp_years_pick_multi_v2"
            )
        with c2:
            metric = st.selectbox(
                "Highlight based on",
                options=["Sales", "Units"],
                index=0,
                key="cmp_multi_metric_v2"
            )

        years_pick = [int(y) for y in years_pick]
        if len(years_pick) < 2:
            st.info("Pick at least two years.")
            return
        years_pick = years_pick[:5]

        dd = d[d["Year"].isin(years_pick)].copy()
        if sel:
            dd = dd[dd[by].isin(sel)]

        # Store for Top SKU movers table: compare first year vs last year in the selection
        try:
            y_first = int(years_pick[0]); y_last = int(years_pick[-1])
            a_df = dd[dd["Year"] == y_last].copy()
            b_df = dd[dd["Year"] == y_first].copy()
            # Context for Explain + One-pager: first vs last year
            a_ctx = dd[dd["Year"] == y_first].copy()
            b_ctx = dd[dd["Year"] == y_last].copy()
            st.session_state["cmp_ctx"] = {"a": a_ctx, "b": b_ctx, "label_a": str(y_first), "label_b": str(y_last), "value_col": metric, "by": by, "filters": {"retailers": f_retailers, "vendors": f_vendors, "limit_by": by, "limit_values": sel}}
            st.session_state["movers_a_periods"] = [str(p) for p in sorted(a_df["MonthP"].unique().tolist())] if "MonthP" in a_df.columns else []
            st.session_state["movers_b_periods"] = [str(p) for p in sorted(b_df["MonthP"].unique().tolist())] if "MonthP" in b_df.columns else []
        except Exception:
            st.session_state["movers_a_periods"] = []
            st.session_state["movers_b_periods"] = []

        pieces = []
        for y in years_pick:
            gy = dd[dd["Year"] == int(y)].groupby(by, as_index=False).agg(**{
                f"Units_{y}": ("Units", "sum"),
                f"Sales_{y}": ("Sales", "sum"),
            })
            pieces.append(gy)

        out = pieces[0]
        for p in pieces[1:]:
            out = out.merge(p, on=by, how="outer")

        out = out.fillna(0.0)

        # Totals row
        total = {by: "TOTAL"}
        for c in out.columns:
            if c == by:
                continue
            total[c] = float(out[c].sum()) if pd.api.types.is_numeric_dtype(out[c]) else ""
        out = pd.concat([out, pd.DataFrame([total])], ignore_index=True)

        # Column order
        cols = [by]
        for y in years_pick:
            cols += [f"Units_{y}", f"Sales_{y}"]
        disp = out[cols].copy()

        # Highlight: highest and lowest across selected years for chosen metric
        metric_cols = [f"{metric}_{y}" for y in years_pick if f"{metric}_{y}" in disp.columns]

        # --- Extra insights (trend, CAGR, sparkline) ---
        # Use the selected metric across the chosen years
        spark_chars = ["▁","▂","▃","▄","▅","▆","▇","█"]

        def _sparkline(vals):
            vals = [float(v) if v is not None and not pd.isna(v) else np.nan for v in vals]
            if len(vals) == 0 or all(pd.isna(v) for v in vals):
                return ""
            vmin = np.nanmin(vals)
            vmax = np.nanmax(vals)
            # all equal -> flat line
            if np.isnan(vmin) or np.isnan(vmax) or np.isclose(vmin, vmax):
                return "▁" * len(vals)
            out_s = []
            for v in vals:
                if pd.isna(v):
                    out_s.append(" ")
                    continue
                t = (v - vmin) / (vmax - vmin)
                idx = int(round(t * (len(spark_chars) - 1)))
                idx = max(0, min(len(spark_chars) - 1, idx))
                out_s.append(spark_chars[idx])
            return "".join(out_s)

        def _pct_change(a, b):
            try:
                a = float(a); b = float(b)
            except Exception:
                return np.nan
            if a == 0:
                return np.nan
            return (b - a) / a

        def _cagr(a, b, periods):
            try:
                a = float(a); b = float(b)
            except Exception:
                return np.nan
            if a <= 0 or b <= 0 or periods <= 0:
                return np.nan
            return (b / a) ** (1.0 / periods) - 1.0

        # Build per-row series for chosen metric (exclude TOTAL row)
        metric_year_cols = [(y, f"{metric}_{y}") for y in years_pick if f"{metric}_{y}" in disp.columns]
        if metric_year_cols:
            series_vals = disp[[c for _, c in metric_year_cols]].copy()

            # Sparkline
            disp["Spark"] = series_vals.apply(lambda r: _sparkline(r.tolist()), axis=1)

            # Trend (first -> last)
            first_col = metric_year_cols[0][1]
            last_col = metric_year_cols[-1][1]
            pct = series_vals.apply(lambda r: _pct_change(r[first_col], r[last_col]), axis=1)
            disp["Trend"] = np.where(
                pct.isna(),
                "—",
                np.where(pct > 0, "↑", np.where(pct < 0, "↓", "→"))
            )
            disp["Trend %"] = pct

            # CAGR across (n_years - 1) intervals
            periods = max(1, len(metric_year_cols) - 1)
            disp["CAGR %"] = series_vals.apply(lambda r: _cagr(r[first_col], r[last_col], periods), axis=1)

            # Clear insight columns on TOTAL row (if present)
            try:
                is_total = disp[by].astype(str) == "TOTAL"
                for c in ["Spark", "Trend", "Trend %", "CAGR %"]:
                    if c in disp.columns:
                        disp.loc[is_total, c] = ""
            except Exception:
                pass

        # Move insight columns next to the year columns
        insight_cols = [c for c in ["Spark", "Trend", "Trend %", "CAGR %"] if c in disp.columns]
        if insight_cols:
            disp = disp[[by] + [c for c in disp.columns if c != by and c not in insight_cols] + insight_cols]
        def _hl_minmax(row):
            styles = [""] * len(row)
            # Don't highlight TOTAL row
            if str(row.iloc[0]) == "TOTAL":
                return styles
            vals = []
            idxs = []
            for j, c in enumerate(disp.columns):
                if c in metric_cols:
                    try:
                        v = float(row[c])
                    except Exception:
                        v = np.nan
                    vals.append(v)
                    idxs.append(j)
            if not vals:
                return styles
            vmin = np.nanmin(vals)
            vmax = np.nanmax(vals)
            # If all equal, no highlight
            if np.isnan(vmin) or np.isnan(vmax) or np.isclose(vmin, vmax):
                return styles
            for v, j in zip(vals, idxs):
                if np.isclose(v, vmax):
                    styles[j] = "background-color: rgba(0, 200, 0, 0.18); font-weight: 600;"
                elif np.isclose(v, vmin):
                    styles[j] = "background-color: rgba(220, 0, 0, 0.14);"
            return styles

        fmt = {}
        for c in disp.columns:
            if c.startswith("Units_"):
                fmt[c] = fmt_int
            elif c.startswith("Sales_"):
                fmt[c] = fmt_currency
        if "Trend %" in disp.columns:
            fmt["Trend %"] = lambda v: (f"{float(v)*100:.1f}%" if (v is not None and pd.notna(v) and str(v).strip() not in {"", "—"}) else "—")
        if "CAGR %" in disp.columns:
            fmt["CAGR %"] = lambda v: (f"{float(v)*100:.1f}%" if (v is not None and pd.notna(v) and str(v).strip() not in {"", "—"}) else "—")

        sty = disp.style.format(fmt).apply(_hl_minmax, axis=1)

        st.dataframe(sty, use_container_width=True, hide_index=True)




def render_comparison_sku():
        st.subheader("SKU Comparison")
        st.session_state["cmp_ctx"] = {}

        if df_all.empty:
            st.info("No sales data yet.")
            return

        d = df_all.copy()
        d["StartDate"] = pd.to_datetime(d["StartDate"], errors="coerce")
        d = d[d["StartDate"].notna()].copy()
        # Optional: select individual SKUs to compare
        if "SKU" in d.columns:
            all_skus = sorted([str(x).strip() for x in d["SKU"].dropna().unique().tolist() if str(x).strip()])
            sku_pick = st.multiselect("Select individual SKUs (optional)", options=all_skus, default=[], key="skucmp_sku_pick")
            if sku_pick:
                d = d[d["SKU"].astype(str).isin([str(s).strip() for s in sku_pick])].copy()


        # Month options across ALL years
        d["MonthP"] = d["StartDate"].dt.to_period("M")
        months = sorted(d["MonthP"].unique().tolist())
        month_labels = [m.to_timestamp().strftime("%B %Y") for m in months]
        label_to_period = dict(zip(month_labels, months))

        d["Year"] = d["StartDate"].dt.year.astype(int)
        years = sorted(d["Year"].dropna().unique().tolist())

        mode = st.radio(
            "Compare type",
            options=["A vs B (Months)", "A vs B (Years)", "Multi-year (high/low highlight)", "Multi-month across years"],
            index=0,
            horizontal=True,
            key="skucmp_mode_v2"
        )

        c1, c2, c3 = st.columns([2, 2, 1])
        with c3:
            filt_by = st.selectbox("Filter by", ["All", "Retailer", "Vendor"], index=0, key="skucmp_filter_by_v2")

        # Optional limiter list
        sel = []
        if filt_by == "Retailer":
            opts = sorted([x for x in d["Retailer"].dropna().unique().tolist() if str(x).strip()])
            sel = st.multiselect("Limit to retailer(s) (optional)", options=opts, key="skucmp_limit_retailer_v2")
        elif filt_by == "Vendor":
            opts = sorted([x for x in d["Vendor"].dropna().unique().tolist() if str(x).strip()])
            sel = st.multiselect("Limit to vendor(s) (optional)", options=opts, key="skucmp_limit_vendor_v2")

        def _apply_filter(dd: pd.DataFrame) -> pd.DataFrame:
            if filt_by == "Retailer" and sel:
                return dd[dd["Retailer"].isin(sel)].copy()
            if filt_by == "Vendor" and sel:
                return dd[dd["Vendor"].isin(sel)].copy()
            return dd

        def _render_a_vs_b(da: pd.DataFrame, db: pd.DataFrame, label_a: str, label_b: str):
            ga = da.groupby("SKU", as_index=False).agg(Units_A=("Units","sum"), Sales_A=("Sales","sum"))
            gb = db.groupby("SKU", as_index=False).agg(Units_B=("Units","sum"), Sales_B=("Sales","sum"))

            out = ga.merge(gb, on="SKU", how="outer").fillna(0.0)
            out["Units_Diff"] = out["Units_A"] - out["Units_B"]
            out["Sales_Diff"] = out["Sales_A"] - out["Sales_B"]
            out["Units_%"] = out["Units_Diff"] / out["Units_B"].replace(0, np.nan)
            out["Sales_%"] = out["Sales_Diff"] / out["Sales_B"].replace(0, np.nan)

            # Add Vendor from map for context
            try:
                if isinstance(vmap, pd.DataFrame) and "SKU" in vmap.columns and "Vendor" in vmap.columns:
                    sku_vendor = vmap[["SKU","Vendor"]].drop_duplicates()
                    out = out.merge(sku_vendor, on="SKU", how="left")
            except Exception:
                pass

            total = {
                "SKU": "TOTAL",
                "Units_A": out["Units_A"].sum(),
                "Sales_A": out["Sales_A"].sum(),
                "Units_B": out["Units_B"].sum(),
                "Sales_B": out["Sales_B"].sum(),
            }
            total["Units_Diff"] = total["Units_A"] - total["Units_B"]
            total["Sales_Diff"] = total["Sales_A"] - total["Sales_B"]
            total["Units_%"] = total["Units_Diff"] / total["Units_B"] if total["Units_B"] else np.nan
            total["Sales_%"] = total["Sales_Diff"] / total["Sales_B"] if total["Sales_B"] else np.nan
            out = pd.concat([out, pd.DataFrame([total])], ignore_index=True)

            cols = ["SKU"] + (["Vendor"] if "Vendor" in out.columns else []) + ["Units_A","Sales_A","Units_B","Sales_B","Units_Diff","Units_%","Sales_Diff","Sales_%"]
            disp = out[cols].copy()
            disp = disp.rename(columns={
                "Units_A": f"Units ({label_a})",
                "Sales_A": f"Sales ({label_a})",
                "Units_B": f"Units ({label_b})",
                "Sales_B": f"Sales ({label_b})",
            })

            sty = disp.style.format({
                f"Units ({label_a})": fmt_int,
                f"Units ({label_b})": fmt_int,
                "Units_Diff": fmt_int_signed,
                "Units_%": lambda v: f"{v*100:.1f}%" if pd.notna(v) else "—",
                f"Sales ({label_a})": fmt_currency,
                f"Sales ({label_b})": fmt_currency,
                "Sales_Diff": fmt_currency_signed,
                "Sales_%": lambda v: f"{v*100:.1f}%" if pd.notna(v) else "—",
            }).applymap(lambda v: f"color: {_color(v)};", subset=["Units_Diff","Sales_Diff"])

            st.dataframe(sty, use_container_width=True, hide_index=True, height=_table_height(disp, max_px=1200))

        # -------------------------
        # Mode 1: A vs B (Months)
        # -------------------------
        if mode == "A vs B (Months)":
            with c1:
                a_pick = st.multiselect(
                    "Selection A (one or more months)",
                    options=month_labels,
                    default=month_labels[-1:] if month_labels else [],
                    key="skucmp_a_months_v2"
                )
            with c2:
                b_pick = st.multiselect(
                    "Selection B (one or more months)",
                    options=month_labels,
                    default=month_labels[-2:-1] if len(month_labels) >= 2 else [],
                    key="skucmp_b_months_v2"
                )

            a_periods = [label_to_period[x] for x in a_pick if x in label_to_period]
            b_periods = [label_to_period[x] for x in b_pick if x in label_to_period]

            st.session_state["movers_a_periods"] = [str(p) for p in a_periods]
            st.session_state["movers_b_periods"] = [str(p) for p in b_periods]

            if not a_periods or not b_periods:
                st.info("Pick at least one month in Selection A and Selection B.")
                return

            da = _apply_filter(d[d["MonthP"].isin(a_periods)])
            db = _apply_filter(d[d["MonthP"].isin(b_periods)])

            label_a = " + ".join(a_pick) if a_pick else "A"
            label_b = " + ".join(b_pick) if b_pick else "B"
            _render_a_vs_b(da, db, label_a, label_b)
            st.session_state["cmp_ctx"] = {"a": da.copy(), "b": db.copy(), "label_a": label_a, "label_b": label_b, "value_col": "Sales"}
            return

        # -------------------------
        # Mode 2: A vs B (Years)
        # -------------------------
        if mode == "A vs B (Years)":
            with c1:
                years_a = st.multiselect(
                    "Selection A (one or more years)",
                    options=years,
                    default=years[-2:-1] if len(years) >= 2 else years,
                    key="skucmp_years_a_v2"
                )
            with c2:
                years_b = st.multiselect(
                    "Selection B (one or more years)",
                    options=years,
                    default=years[-1:] if years else [],
                    key="skucmp_years_b_v2"
                )

            if not years_a or not years_b:
                st.info("Pick at least one year in Selection A and Selection B.")
                return

            da = _apply_filter(d[d["Year"].isin([int(y) for y in years_a])])
            db = _apply_filter(d[d["Year"].isin([int(y) for y in years_b])])

            try:
                st.session_state["movers_a_periods"] = [str(p) for p in sorted(da["MonthP"].unique().tolist())]
                st.session_state["movers_b_periods"] = [str(p) for p in sorted(db["MonthP"].unique().tolist())]
            except Exception:
                st.session_state["movers_a_periods"] = []
                st.session_state["movers_b_periods"] = []

            label_a = " + ".join([str(y) for y in years_a])
            label_b = " + ".join([str(y) for y in years_b])
            _render_a_vs_b(da, db, label_a, label_b)
            st.session_state["cmp_ctx"] = {"a": da.copy(), "b": db.copy(), "label_a": label_a, "label_b": label_b, "value_col": "Sales"}
            return


        # -------------------------
        # Mode 2b: Multi-month across years (pick Month+Year periods)
        # -------------------------
        if mode == "Multi-month across years":
            month_names = ["January","February","March","April","May","June","July","August","September","October","November","December"]
            period_options = []
            for y in years:
                for mname in month_names:
                    period_options.append(f"{mname} {y}")

            with c1:
                periods_pick = st.multiselect(
                    "Month + Year periods",
                    options=period_options,
                    default=[period_options[-1]] if period_options else [],
                    key="skucmp_mm_periods",
                )
            with c2:
                metric = st.selectbox(
                    "Highlight based on",
                    options=["Sales", "Units"],
                    index=0,
                    key="skucmp_mm_metric",
                )
            with c3:
                topn = st.selectbox("Show", options=[25, 50, 100, 250], index=1, key="skucmp_mm_topn")

            if not periods_pick:
                st.info("Pick at least one Month + Year period.")
                return

            month_to_num = {m:i+1 for i,m in enumerate(month_names)}
            pairs = []
            for p in periods_pick:
                try:
                    parts = p.split(" ")
                    mname = " ".join(parts[:-1])
                    yy = int(parts[-1])
                    mn = month_to_num.get(mname, None)
                    if mn is not None:
                        pairs.append((yy, mn, p))
                except Exception:
                    continue
            if not pairs:
                st.info("No valid periods selected.")
                return

            dd = d.copy()
            mask = False
            for (yy, mn, _) in pairs:
                mask = mask | ((dd["Year"] == int(yy)) & (dd["StartDate"].dt.month == int(mn)))
            dd = dd[mask].copy()

            if dd.empty:
                st.info("No data found for those periods with current filters.")
                return

            pieces = []
            for (yy, mn, lab_full) in pairs:
                lab = lab_full.replace("January","Jan").replace("February","Feb").replace("March","Mar").replace("April","Apr").replace("June","Jun").replace("July","Jul").replace("August","Aug").replace("September","Sep").replace("October","Oct").replace("November","Nov").replace("December","Dec")
                dyy = dd[(dd["Year"] == int(yy)) & (dd["StartDate"].dt.month == int(mn))].copy()
                gy = dyy.groupby("SKU", as_index=False).agg(**{
                    f"Units_{lab}": ("Units", "sum"),
                    f"Sales_{lab}": ("Sales", "sum"),
                })
                pieces.append((lab, gy))

            out = pieces[0][1]
            for lab, p in pieces[1:]:
                out = out.merge(p, on="SKU", how="outer")
            out = out.fillna(0.0)

            total = {"SKU": "TOTAL"}
            for c in out.columns:
                if c == "SKU":
                    continue
                total[c] = float(out[c].sum()) if pd.api.types.is_numeric_dtype(out[c]) else ""
            out = pd.concat([out, pd.DataFrame([total])], ignore_index=True)

            cols = ["SKU"]
            for lab, _ in pieces:
                cols += [f"Units_{lab}", f"Sales_{lab}"]
            disp = out[cols].copy()

            metric_cols = [c for c in disp.columns if c.startswith(metric + "_")]
            if metric_cols:
                disp = disp.sort_values(metric_cols[-1], ascending=False).head(int(topn)).copy()

            spark_chars = ["▁","▂","▃","▄","▅","▆","▇","█"]
            def _spark(vals):
                vals = [float(v) if v is not None and not pd.isna(v) else np.nan for v in vals]
                if len(vals) == 0 or all(pd.isna(v) for v in vals):
                    return ""
                vmin = np.nanmin(vals); vmax = np.nanmax(vals)
                if np.isnan(vmin) or np.isnan(vmax) or np.isclose(vmin, vmax):
                    return "▁" * len(vals)
                out_s=[]
                for v in vals:
                    if pd.isna(v):
                        out_s.append(" "); continue
                    t=(v-vmin)/(vmax-vmin)
                    idx=int(round(t*(len(spark_chars)-1)))
                    idx=max(0,min(len(spark_chars)-1,idx))
                    out_s.append(spark_chars[idx])
                return "".join(out_s)

            def _cagr(a,b,periods):
                try:
                    a=float(a); b=float(b)
                except Exception:
                    return np.nan
                if a <= 0 or b <= 0 or periods <= 0:
                    return np.nan
                return (b/a)**(1.0/periods)-1.0

            if metric_cols:
                series_vals = disp[metric_cols].copy()
                disp["Spark"] = series_vals.apply(lambda r: _spark(r.tolist()), axis=1)
                periods_n = max(1, len(metric_cols)-1)
                disp["CAGR %"] = series_vals.apply(lambda r: _cagr(r[metric_cols[0]], r[metric_cols[-1]], periods_n), axis=1)

            def _hl(row):
                styles=[""]*len(row)
                if not metric_cols:
                    return styles
                vals=[]
                for c in metric_cols:
                    try:
                        vals.append(float(row[c]))
                    except Exception:
                        vals.append(np.nan)
                if len(vals)==0 or all(pd.isna(v) for v in vals):
                    return styles
                vmax=np.nanmax(vals); vmin=np.nanmin(vals)
                for i,c in enumerate(row.index):
                    if c in metric_cols:
                        v=float(row[c]) if pd.notna(row[c]) else np.nan
                        if pd.notna(v) and np.isclose(v,vmax):
                            styles[i]="background-color: rgba(0, 200, 0, 0.18); font-weight: 600;"
                        elif pd.notna(v) and np.isclose(v,vmin):
                            styles[i]="background-color: rgba(220, 0, 0, 0.14);"
                return styles

            fmt_map = {c: fmt_int for c in disp.columns if c.startswith("Units_")}
            fmt_map.update({c: fmt_currency for c in disp.columns if c.startswith("Sales_")})
            if "CAGR %" in disp.columns:
                fmt_map["CAGR %"] = lambda v: f"{v*100:.1f}%" if pd.notna(v) else "—"

            st.caption("Multi-month across years (selected Month+Year periods)")
            sty = disp.style.format(fmt_map)
            if metric_cols:
                sty = sty.apply(_hl, axis=1)
            st.dataframe(sty, use_container_width=True, hide_index=True)

            first = pairs[0]; last = pairs[-1]
            a_ctx = dd[(dd["Year"] == int(first[0])) & (dd["StartDate"].dt.month == int(first[1]))].copy()
            b_ctx = dd[(dd["Year"] == int(last[0])) & (dd["StartDate"].dt.month == int(last[1]))].copy()
            st.session_state["cmp_ctx"] = {"a": a_ctx, "b": b_ctx, "label_a": first[2], "label_b": last[2], "value_col": metric}
            return

        # -------------------------
        # Mode 3: Multi-year highlight table
        # -------------------------
        with c1:
            years_pick = st.multiselect(
                "Years to view (pick 2 to 5)",
                options=years,
                default=years[-3:] if len(years) >= 3 else years,
                key="skucmp_years_pick_multi_v2"
            )
        with c2:
            metric = st.selectbox(
                "Highlight based on",
                options=["Sales", "Units"],
                index=0,
                key="skucmp_multi_metric_v2"
            )

        years_pick = [int(y) for y in years_pick]
        if len(years_pick) < 2:
            st.info("Pick at least two years.")
            return
        years_pick = years_pick[:5]

        dd = _apply_filter(d[d["Year"].isin(years_pick)].copy())

        # Movers compare first vs last selected year
        try:
            y_first = int(years_pick[0]); y_last = int(years_pick[-1])
            a_df = dd[dd["Year"] == y_last].copy()
            b_df = dd[dd["Year"] == y_first].copy()
            # Context for Explain + One-pager: first vs last year
            a_ctx = dd[dd["Year"] == y_first].copy()
            b_ctx = dd[dd["Year"] == y_last].copy()
            st.session_state["cmp_ctx"] = {"a": a_ctx, "b": b_ctx, "label_a": str(y_first), "label_b": str(y_last), "value_col": metric}
            st.session_state["movers_a_periods"] = [str(p) for p in sorted(a_df["MonthP"].unique().tolist())]
            st.session_state["movers_b_periods"] = [str(p) for p in sorted(b_df["MonthP"].unique().tolist())]
        except Exception:
            st.session_state["movers_a_periods"] = []
            st.session_state["movers_b_periods"] = []

        pieces = []
        for y in years_pick:
            gy = dd[dd["Year"] == int(y)].groupby("SKU", as_index=False).agg(**{
                f"Units_{y}": ("Units", "sum"),
                f"Sales_{y}": ("Sales", "sum"),
            })
            pieces.append(gy)

        out = pieces[0]
        for p in pieces[1:]:
            out = out.merge(p, on="SKU", how="outer")

        out = out.fillna(0.0)

        # Totals row
        total = {"SKU": "TOTAL"}
        for c in out.columns:
            if c == "SKU":
                continue
            total[c] = float(out[c].sum()) if pd.api.types.is_numeric_dtype(out[c]) else ""
        out = pd.concat([out, pd.DataFrame([total])], ignore_index=True)

        cols = ["SKU"]
        for y in years_pick:
            cols += [f"Units_{y}", f"Sales_{y}"]
        disp = out[cols].copy()

        metric_cols = [f"{metric}_{y}" for y in years_pick if f"{metric}_{y}" in disp.columns]

        def _hl_minmax(row):
            styles = [""] * len(row)
            if str(row.iloc[0]) == "TOTAL":
                return styles
            vals = []
            idxs = []
            for j, c in enumerate(disp.columns):
                if c in metric_cols:
                    try:
                        v = float(row[c])
                    except Exception:
                        v = np.nan
                    vals.append(v)
                    idxs.append(j)
            if not vals:
                return styles
            vmin = np.nanmin(vals)
            vmax = np.nanmax(vals)
            if np.isnan(vmin) or np.isnan(vmax) or np.isclose(vmin, vmax):
                return styles
            for v, j in zip(vals, idxs):
                if np.isclose(v, vmax):
                    styles[j] = "background-color: rgba(0, 200, 0, 0.18); font-weight: 600;"
                elif np.isclose(v, vmin):
                    styles[j] = "background-color: rgba(220, 0, 0, 0.14);"
            return styles

        fmt = {}
        for c in disp.columns:
            if c.startswith("Units_"):
                fmt[c] = fmt_int
            elif c.startswith("Sales_"):
                fmt[c] = fmt_currency

        st.dataframe(disp.style.format(fmt).apply(_hl_minmax, axis=1), use_container_width=True, hide_index=True, height=_table_height(disp, max_px=1200))


def render_sku_health():
        st.subheader("SKU Health Score")

        if df_all.empty:
            st.info("No sales data yet.")
            return

        d = df_all.copy()
        d["StartDate"] = pd.to_datetime(d["StartDate"], errors="coerce")
        d = d[d["StartDate"].notna()].copy()

        d["Year"] = d["StartDate"].dt.year.astype(int)
        d["Month"] = d["StartDate"].dt.month.astype(int)
        d["MonthP"] = d["StartDate"].dt.to_period("M")

        compare_mode = st.selectbox(
            "Compare mode",
            options=["Year vs Year", "Month vs Month (multi-month)"],
            index=0,
            key="sh_compare_mode"
        )

        basis = st.radio("Primary basis", options=["Sales", "Units"], index=0, horizontal=True, key="sh_basis")

        # Shared filters
        f1, f2, f3, f4 = st.columns([2, 2, 1, 1])
        with f1:
            vendor_filter = st.multiselect(
                "Vendor filter (optional)",
                options=sorted([x for x in d["Vendor"].dropna().unique().tolist() if str(x).strip()]),
                key="sh_vendor_filter"
            )
        with f2:
            retailer_filter = st.multiselect(
                "Retailer filter (optional)",
                options=sorted([x for x in d["Retailer"].dropna().unique().tolist() if str(x).strip()]),
                key="sh_retailer_filter"
            )
        with f3:
            top_n = st.number_input("Top N", min_value=20, max_value=2000, value=200, step=20, key="sh_topn")
        with f4:
            status_pick = st.multiselect(
                "Status",
                options=["🔥 Strong","📈 Growing","⚠ Watch","❌ At Risk"],
                default=["🔥 Strong","📈 Growing","⚠ Watch","❌ At Risk"],
                key="sh_status"
            )

        dd = d.copy()
        if vendor_filter:
            dd = dd[dd["Vendor"].isin(vendor_filter)]
        if retailer_filter:
            dd = dd[dd["Retailer"].isin(retailer_filter)]

        # Build A vs B selections
        if compare_mode == "Year vs Year":
            years = sorted(dd["Year"].unique().tolist())
            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                base_year = st.selectbox("Base year", options=years, index=max(0, len(years)-2), key="sh_base")
            with c2:
                comp_year = st.selectbox("Compare to", options=years, index=len(years)-1 if years else 0, key="sh_comp")
            with c3:
                pmode = st.selectbox("Period", options=["Full year", "Specific months"], index=0, key="sh_period_mode")

            sel_months = list(range(1,13))
            if pmode == "Specific months":
                month_name = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
                month_list = [month_name[i] for i in range(1,13)]
                sel_names = st.multiselect("Months", options=month_list, default=[month_list[0]], key="sh_months_pick")
                sel_months = [k for k,v in month_name.items() if v in sel_names]

            a = dd[(dd["Year"] == int(base_year)) & (dd["Month"].isin(sel_months))].copy()
            b = dd[(dd["Year"] == int(comp_year)) & (dd["Month"].isin(sel_months))].copy()

            a_label = str(base_year)
            b_label = str(comp_year)

        else:
            # Month vs Month (can be same year or different years)
            months = sorted(dd["MonthP"].unique().tolist())
            month_labels = [m.to_timestamp().strftime("%B %Y") for m in months]
            label_to_period = dict(zip(month_labels, months))

            c1, c2 = st.columns(2)
            with c1:
                a_pick = st.multiselect(
                    "Selection A (one or more months)",
                    options=month_labels,
                    default=month_labels[-1:] if month_labels else [],
                    key="sh_mm_a"
                )
            with c2:
                b_pick = st.multiselect(
                    "Selection B (one or more months)",
                    options=month_labels,
                    default=month_labels[-2:-1] if len(month_labels) >= 2 else [],
                    key="sh_mm_b"
                )

            a_periods = [label_to_period[x] for x in a_pick if x in label_to_period]
            b_periods = [label_to_period[x] for x in b_pick if x in label_to_period]

            if (not a_periods) or (not b_periods):
                st.info("Pick at least one month in Selection A and Selection B.")
                return

            a = dd[dd["MonthP"].isin(a_periods)].copy()
            b = dd[dd["MonthP"].isin(b_periods)].copy()

            a_label = "Selection A"
            b_label = "Selection B"

        ga = a.groupby("SKU", as_index=False).agg(Sales_A=("Sales","sum"), Units_A=("Units","sum"))
        gb = b.groupby("SKU", as_index=False).agg(Sales_B=("Sales","sum"), Units_B=("Units","sum"))
        out = ga.merge(gb, on="SKU", how="outer").fillna(0.0)

        # Coverage context (based on B selection)
        cov = b.groupby("SKU", as_index=False).agg(Retailers=("Retailer","nunique"), ActiveWeeks=("StartDate","nunique"))
        out = out.merge(cov, on="SKU", how="left").fillna({"Retailers": 0, "ActiveWeeks": 0})

        out["Δ Sales"] = out["Sales_B"] - out["Sales_A"]
        out["Δ Units"] = out["Units_B"] - out["Units_A"]
        out["Sales %"] = out["Δ Sales"] / out["Sales_A"].replace(0, np.nan)
        out["Units %"] = out["Δ Units"] / out["Units_A"].replace(0, np.nan)

        out["Score"] = out["Δ Sales"] if basis == "Sales" else out["Δ Units"]

        def _status(row):
            a0 = float(row["Sales_A"] if basis=="Sales" else row["Units_A"])
            b0 = float(row["Sales_B"] if basis=="Sales" else row["Units_B"])
            delta = b0 - a0
            if a0 == 0 and b0 > 0:
                return "📈 Growing"
            if a0 > 0 and b0 == 0:
                return "❌ At Risk"
            if delta > 0:
                return "🔥 Strong"
            if delta < 0:
                return "⚠ Watch"
            return "⚠ Watch"

        out["Status"] = out.apply(_status, axis=1)
        out = out[out["Status"].isin(status_pick)].copy()
        out = out.sort_values("Score", ascending=False, kind="mergesort").head(int(top_n))

        # Vendor lookup
        try:
            if isinstance(vmap, pd.DataFrame) and "SKU" in vmap.columns and "Vendor" in vmap.columns:
                out = out.merge(vmap[["SKU","Vendor"]].drop_duplicates(), on="SKU", how="left")
        except Exception:
            pass

        cols = ["SKU"] + (["Vendor"] if "Vendor" in out.columns else []) + ["Status","Sales_A","Sales_B","Δ Sales","Sales %","Units_A","Units_B","Δ Units","Units %","Retailers","ActiveWeeks"]
        disp = out[cols].copy()
        disp = disp.rename(columns={
            "Sales_A": a_label,
            "Sales_B": b_label,
            "Units_A": f"Units {a_label}",
            "Units_B": f"Units {b_label}",
        })
        disp = make_unique_columns(disp)

        st.dataframe(
            disp.style.format({
                a_label: fmt_currency,
                b_label: fmt_currency,
                "Δ Sales": fmt_currency,
                "Sales %": lambda v: f"{v*100:.1f}%" if pd.notna(v) else "—",
                f"Units {a_label}": fmt_int,
                f"Units {b_label}": fmt_int,
                "Δ Units": fmt_int,
                "Units %": lambda v: f"{v*100:.1f}%" if pd.notna(v) else "—",
                "Retailers": fmt_int,
                "ActiveWeeks": fmt_int,
            }),
            use_container_width=True,
            hide_index=True,
            height=_table_height(disp, max_px=1200)
        )

def render_lost_sales():
        st.subheader("Lost Sales Detector")

        if df_all.empty:
            st.info("No sales data yet.")
            return

        d = df_all.copy()
        d["StartDate"] = pd.to_datetime(d["StartDate"], errors="coerce")
        d = d[d["StartDate"].notna()].copy()
        d["Year"] = d["StartDate"].dt.year.astype(int)
        d["Month"] = d["StartDate"].dt.month.astype(int)

        years = sorted(d["Year"].unique().tolist())
        month_name = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        month_list = [month_name[i] for i in range(1,13)]

        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            base_year = st.selectbox("Base year", options=years, index=max(0, len(years)-2), key="ls_base")
        with c2:
            comp_year = st.selectbox("Compare to", options=years, index=len(years)-1, key="ls_comp")
        with c3:
            basis = st.radio("Basis", options=["Sales", "Units"], index=0, horizontal=True, key="ls_basis")

        pmode = st.selectbox("Period", options=["Full year", "Specific months"], index=0, key="ls_period_mode")
        sel_months = list(range(1,13))
        if pmode == "Specific months":
            sel_names = st.multiselect("Months", options=month_list, default=[month_list[0]], key="ls_months_pick")
            sel_months = [k for k,v in month_name.items() if v in sel_names]

        value_col = "Sales" if basis == "Sales" else "Units"

        a = d[(d["Year"] == int(base_year)) & (d["Month"].isin(sel_months))].copy()
        b = d[(d["Year"] == int(comp_year)) & (d["Month"].isin(sel_months))].copy()

        # Build SKU-level totals for both Units and Sales (for summary + gained/lost)
        ga_all = a.groupby("SKU", as_index=False).agg(Units_A=("Units","sum"), Sales_A=("Sales","sum"))
        gb_all = b.groupby("SKU", as_index=False).agg(Units_B=("Units","sum"), Sales_B=("Sales","sum"))
        sku_all = ga_all.merge(gb_all, on="SKU", how="outer").fillna(0.0)

        lost_mask = ((sku_all["Units_A"] > 0) | (sku_all["Sales_A"] > 0)) & (sku_all["Units_B"] == 0) & (sku_all["Sales_B"] == 0)
        gained_mask = (sku_all["Units_A"] == 0) & (sku_all["Sales_A"] == 0) & ((sku_all["Units_B"] > 0) | (sku_all["Sales_B"] > 0))

        lost_all = sku_all[lost_mask].copy()
        gained_all = sku_all[gained_mask].copy()

        # Summary totals
        lost_units = float(lost_all["Units_A"].sum()) if not lost_all.empty else 0.0
        lost_sales = float(lost_all["Sales_A"].sum()) if not lost_all.empty else 0.0
        gained_units = float(gained_all["Units_B"].sum()) if not gained_all.empty else 0.0
        gained_sales = float(gained_all["Sales_B"].sum()) if not gained_all.empty else 0.0

        net_units = gained_units - lost_units
        net_sales = gained_sales - lost_sales

        # Show net impact (green if net positive, red if net negative)
        net_color = "#2ecc71" if net_sales > 0 else ("#e74c3c" if net_sales < 0 else "#999999")
        st.markdown(
            f"""<div style="padding:10px 12px; border-radius:10px; border:1px solid #2a2a2a;">
            <div style="font-size:0.95rem; font-weight:700;">Net change (Gained − Lost)</div>
            <div style="margin-top:6px; color:{net_color}; font-weight:800; font-size:1.15rem;">
              {fmt_currency(net_sales)} &nbsp;|&nbsp; {fmt_int(net_units)} units
            </div>
            <div style="margin-top:6px; font-size:0.9rem; color:#aaaaaa;">
              Lost: {fmt_currency(lost_sales)} / {fmt_int(lost_units)} units &nbsp;&nbsp;•&nbsp;&nbsp;
              Gained: {fmt_currency(gained_sales)} / {fmt_int(gained_units)} units
            </div>
            </div>""",
            unsafe_allow_html=True
        )

        # Basis-specific (existing behavior)
        ga = a.groupby("SKU", as_index=False).agg(A=(value_col,"sum"))
        gb = b.groupby("SKU", as_index=False).agg(B=(value_col,"sum"))
        sku = ga.merge(gb, on="SKU", how="outer").fillna(0.0)
        sku["Delta"] = sku["B"] - sku["A"]
        sku["Pct"] = sku["Delta"] / sku["A"].replace(0, np.nan)

        lost = sku[(sku["A"] > 0) & (sku["B"] == 0)].copy().sort_values("A", ascending=False).head(200)
        gained = sku[(sku["A"] == 0) & (sku["B"] > 0)].copy().sort_values("B", ascending=False).head(200)
        drops = sku[(sku["A"] > 0) & (sku["B"] > 0) & (sku["Delta"] < 0)].copy().sort_values("Delta").head(200)

        ra = a.groupby(["SKU","Retailer"], as_index=False).agg(A=(value_col,"sum"))
        rb = b.groupby(["SKU","Retailer"], as_index=False).agg(B=(value_col,"sum"))
        rr = ra.merge(rb, on=["SKU","Retailer"], how="outer").fillna(0.0)
        rr["Delta"] = rr["B"] - rr["A"]
        lost_retail = rr[(rr["A"] > 0) & (rr["B"] == 0)].copy().sort_values("A", ascending=False).head(300)

        try:
            if isinstance(vmap, pd.DataFrame) and "SKU" in vmap.columns and "Vendor" in vmap.columns:
                vend = vmap[["SKU","Vendor"]].drop_duplicates()
                lost = lost.merge(vend, on="SKU", how="left")
                drops = drops.merge(vend, on="SKU", how="left")
                lost_retail = lost_retail.merge(vend, on="SKU", how="left")
        except Exception:
            pass

        def _fmt(v):
            return fmt_currency(v) if value_col == "Sales" else fmt_int(v)

        st.markdown("### Lost SKUs (sold in base period, zero in compare period)")
        lost_disp = lost[["SKU"] + (["Vendor"] if "Vendor" in lost.columns else []) + ["A"]].copy().rename(columns={"A": str(base_year)})
        lost_disp = make_unique_columns(lost_disp)
        st.dataframe(lost_disp.style.format({str(base_year): _fmt}), use_container_width=True, hide_index=True, height=650)

        st.markdown("### Gained SKUs (zero in base period, sold in compare period)")
        gained_disp = gained[["SKU"] + (["Vendor"] if "Vendor" in gained.columns else []) + ["B"]].copy().rename(columns={"B": str(comp_year)})
        gained_disp = make_unique_columns(gained_disp)
        st.dataframe(gained_disp.style.format({str(comp_year): _fmt}), use_container_width=True, hide_index=True, height=650)

        st.markdown("### Biggest declines (still selling, but down)")
        drops_disp = drops[["SKU"] + (["Vendor"] if "Vendor" in drops.columns else []) + ["A","B","Delta","Pct"]].copy()
        drops_disp = drops_disp.rename(columns={"A": str(base_year), "B": str(comp_year)})
        drops_disp = make_unique_columns(drops_disp)
        st.dataframe(
            drops_disp.style.format({
                str(base_year): _fmt,
                str(comp_year): _fmt,
                "Delta": _fmt,
                "Pct": lambda v: f"{v*100:.1f}%" if pd.notna(v) else "—",
            }),
            use_container_width=True,
            hide_index=True,
            height=650
        )

        st.markdown("### Lost retailers for specific SKUs")
        lost_retail_disp = lost_retail[["SKU","Retailer"] + (["Vendor"] if "Vendor" in lost_retail.columns else []) + ["A"]].copy()
        lost_retail_disp = lost_retail_disp.rename(columns={"A": str(base_year)})
        lost_retail_disp = make_unique_columns(lost_retail_disp)
        st.dataframe(lost_retail_disp.style.format({str(base_year): _fmt}), use_container_width=True, hide_index=True, height=700)

def render_data_inventory():
        st.subheader("Data Inventory")

        if df_all.empty:
            st.info("No sales data yet.")
        else:
            d = df_all.copy()
            d["StartDate"] = pd.to_datetime(d["StartDate"], errors="coerce")
            d = d[d["StartDate"].notna()].copy()
            d["Year"] = d["StartDate"].dt.year.astype(int)

            st.markdown("### By year")
            by_year = d.groupby("Year", as_index=False).agg(
                Units=("Units","sum"),
                Sales=("Sales","sum"),
                Retailers=("Retailer","nunique"),
                Vendors=("Vendor","nunique"),
                SKUs=("SKU","nunique"),
            ).sort_values("Year", ascending=False)
            st.dataframe(by_year.style.format({
                "Units": fmt_int, "Sales": fmt_currency,
                "Retailers": fmt_int, "Vendors": fmt_int, "SKUs": fmt_int
            }), use_container_width=True, hide_index=True)

            st.markdown("### By retailer (selected year)")
            years = sorted(d["Year"].unique().tolist())
            sel_y = st.selectbox("Year", options=years, index=len(years)-1, key="inv_year")
            dy = d[d["Year"] == int(sel_y)].copy()
            if "SourceFile" not in dy.columns:
                dy["SourceFile"] = ""
            by_ret = dy.groupby("Retailer", as_index=False).agg(
                Units=("Units","sum"),
                Sales=("Sales","sum"),
                SKUs=("SKU","nunique"),
                Sources=("SourceFile","nunique"),
            ).sort_values("Sales", ascending=False)
            st.dataframe(by_ret.style.format({
                "Units": fmt_int, "Sales": fmt_currency, "SKUs": fmt_int, "Sources": fmt_int
            }), use_container_width=True, height=_table_height(by_ret, max_px=900), hide_index=True)

            st.markdown("### By source file (selected year)")
            by_src = dy.groupby("SourceFile", as_index=False).agg(
                Units=("Units","sum"),
                Sales=("Sales","sum"),
                Retailers=("Retailer","nunique"),
                SKUs=("SKU","nunique"),
            ).sort_values("Sales", ascending=False)
            st.dataframe(by_src.style.format({
                "Units": fmt_int, "Sales": fmt_currency, "Retailers": fmt_int, "SKUs": fmt_int
            }), use_container_width=True, height=_table_height(by_src, max_px=900), hide_index=True)




    # -------------------------
    # Insights & Alerts
    # -------------------------


def render_edit_vendor_map():
        st.subheader("Edit Vendor Map")
        st.caption("Edit Vendor and Price. Click Save to update the default vendor map file used by the app.")
        vmap_disp = vmap[["Retailer","SKU","Vendor","Price","MapOrder"]].copy().sort_values(["Retailer","MapOrder"])
        show = vmap_disp.drop(columns=["MapOrder"]).copy()

        if edit_mode:
            edited = st.data_editor(show, use_container_width=True, hide_index=True, num_rows="dynamic")
            if st.button("Save Vendor Map"):
                updated = edited.copy()
                updated["Retailer"] = updated["Retailer"].map(_normalize_retailer)
                updated["SKU"] = updated["SKU"].map(_normalize_sku)
                updated["Vendor"] = updated["Vendor"].astype(str).str.strip()
                updated["Price"] = pd.to_numeric(updated["Price"], errors="coerce")

                # MapOrder based on current row order per retailer
                updated["MapOrder"] = 0
                for r, grp in updated.groupby("Retailer", sort=False):
                    for j, ix in enumerate(grp.index.tolist()):
                        updated.loc[ix, "MapOrder"] = j

                updated.to_excel(DEFAULT_VENDOR_MAP, index=False)
                st.success("Saved vendor map. Reloading…")
                st.rerun()
        else:
            st.info("Turn on Edit Mode in the sidebar to edit.")
            st.dataframe(show, use_container_width=True, height=_table_height(show, max_px=1400), hide_index=True)

    # Backup / Restore


def render_backup_restore():
        st.subheader("Backup / Restore")

        st.markdown("### Backup files")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("#### Sales database")
            if DEFAULT_SALES_STORE.exists():
                st.download_button("Download sales_store.csv", data=DEFAULT_SALES_STORE.read_bytes(), file_name="sales_store.csv", mime="text/csv")
            else:
                st.info("No sales_store.csv yet.")

            up = st.file_uploader("Restore sales_store.csv", type=["csv"], key="restore_sales_csv")
            if st.button("Restore sales_store.csv", disabled=up is None, key="btn_restore_sales"):
                DEFAULT_SALES_STORE.write_bytes(up.getbuffer())
                st.success("Restored sales_store.csv. Reloading…")
                st.rerun()

        with c2:
            st.markdown("#### Vendor map")
            if DEFAULT_VENDOR_MAP.exists():
                st.download_button("Download vendor_map.xlsx", data=DEFAULT_VENDOR_MAP.read_bytes(), file_name="vendor_map.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                st.info("No vendor_map.xlsx yet.")

            up2 = st.file_uploader("Restore vendor_map.xlsx", type=["xlsx"], key="restore_vm_xlsx")
            if st.button("Restore vendor_map.xlsx", disabled=up2 is None, key="btn_restore_vm"):
                DEFAULT_VENDOR_MAP.write_bytes(up2.getbuffer())
                st.success("Restored vendor_map.xlsx. Reloading…")
                st.rerun()

        with c3:
            st.markdown("#### Price history")
            if DEFAULT_PRICE_HISTORY.exists():
                st.download_button("Download price_history.csv", data=DEFAULT_PRICE_HISTORY.read_bytes(), file_name="price_history.csv", mime="text/csv")
            else:
                st.info("No price_history.csv yet.")

            up3 = st.file_uploader("Restore price_history.csv", type=["csv"], key="restore_ph_csv")
            if st.button("Restore price_history.csv", disabled=up3 is None, key="btn_restore_ph"):
                DEFAULT_PRICE_HISTORY.write_bytes(up3.getbuffer())
                st.success("Restored price_history.csv. Reloading…")
                st.rerun()

        st.markdown("#### Year locks")
        if DEFAULT_YEAR_LOCKS.exists():
            st.download_button("Download year_locks.json", data=DEFAULT_YEAR_LOCKS.read_bytes(), file_name="year_locks.json", mime="application/json")
        else:
            st.info("No year locks saved yet.")

        up4 = st.file_uploader("Restore year_locks.json", type=["json"], key="restore_year_locks")
        if st.button("Restore year_locks.json", disabled=up4 is None, key="btn_restore_year_locks"):
            DEFAULT_YEAR_LOCKS.write_bytes(up4.getbuffer())
            st.success("Restored year locks. Reloading…")
            st.rerun()

        st.divider()

        st.markdown("### Price changes (effective date)")
        st.caption("Upload a sheet with SKU + Price + StartDate. Optional Retailer column. Prices apply from StartDate forward and never change earlier weeks.")

        tmpl = pd.DataFrame([
            {"Retailer":"*", "SKU":"ABC123", "Price": 19.99, "StartDate":"2026-02-01"},
            {"Retailer":"home depot", "SKU":"XYZ999", "Price": 24.99, "StartDate":"2026-03-15"},
        ])
        st.download_button("Download template CSV", data=tmpl.to_csv(index=False).encode("utf-8"),
                           file_name="price_history_template.csv", mime="text/csv")

        ph_up = st.file_uploader("Upload price history (CSV or Excel)", type=["csv","xlsx"], key="ph_upload")
        if ph_up is not None:
            try:
                if ph_up.name.lower().endswith(".csv"):
                    ph_new = pd.read_csv(ph_up)
                else:
                    ph_new = pd.read_excel(ph_up)

                st.markdown("#### Preview upload")
                st.dataframe(ph_new.head(50), use_container_width=True, hide_index=True)

                # Normalize + ignore blanks safely
                cur_ph = load_price_history()
                incoming, ignored = _prepare_price_history_upload(ph_new)
                diff = _price_history_diff(cur_ph, incoming)

                st.divider()
                st.markdown("#### What will change")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Rows uploaded", int(len(ph_new)))
                c2.metric("Rows ignored (blank/invalid)", int(len(ignored)))
                c3.metric("Inserts", int((diff["Action"] == "insert").sum()) if not diff.empty else 0)
                c4.metric("Updates", int((diff["Action"] == "update").sum()) if not diff.empty else 0)

                show_diff = diff.copy()
                if not show_diff.empty:
                    show_diff["StartDate"] = pd.to_datetime(show_diff["StartDate"], errors="coerce").dt.date
                    sty = show_diff.style.format({
                        "OldPrice": lambda v: fmt_currency(v) if pd.notna(v) else "—",
                        "Price": lambda v: fmt_currency(v),
                        "PriceDiff": lambda v: fmt_currency(v) if pd.notna(v) else "—",
                    }).applymap(lambda v: "font-weight:700;" if str(v) in ["insert","update"] else "", subset=["Action"])
                    st.dataframe(sty, use_container_width=True, height=_table_height(show_diff, max_px=900), hide_index=True)
                    st.download_button("Download change preview (CSV)", data=show_diff.to_csv(index=False).encode("utf-8"),
                    file_name="price_history_changes_preview.csv", mime="text/csv")
                else:
                    st.info("No valid rows found in this upload (all prices were blank/invalid).")

                if not ignored.empty:
                    st.markdown("#### Ignored rows")
                    ign = ignored.copy()
                    ign["StartDate"] = pd.to_datetime(ign["StartDate"], errors="coerce").dt.date
                    st.dataframe(ign.head(200), use_container_width=True, height=_table_height(ign, max_px=600), hide_index=True)
                st.download_button("Download ignored rows (CSV)", data=ign.to_csv(index=False).encode("utf-8"),
                    file_name="price_history_ignored_rows.csv", mime="text/csv")
                    
                if st.button("Apply price changes", key="btn_apply_prices"):
                    ins, upd, noop = upsert_price_history(ph_new)
                    st.success(f"Price history updated. Inserts: {ins}, Updates: {upd}, Unchanged: {noop}. Reloading…")
                    st.rerun()
            except Exception as e:
                st.error(f"Could not read this file: {e}")

        if DEFAULT_PRICE_HISTORY.exists():
            if st.button("Clear ALL price history", key="btn_clear_ph"):
                DEFAULT_PRICE_HISTORY.unlink(missing_ok=True)
                st.success("Cleared. Reloading…")
                st.rerun()

        st.divider()

        st.markdown("### Export enriched sales")
        if not df.empty:
            ex = df.copy()
            ex["StartDate"] = pd.to_datetime(ex["StartDate"], errors="coerce").dt.strftime("%Y-%m-%d")
            ex["EndDate"] = pd.to_datetime(ex["EndDate"], errors="coerce").dt.strftime("%Y-%m-%d")
            st.download_button("Download enriched_sales.csv", data=ex.to_csv(index=False).encode("utf-8"),
                               file_name="enriched_sales.csv", mime="text/csv")
        else:
            st.info("No sales yet.")



    # -------------------------
    # Bulk Data Upload
    # -------------------------


def render_bulk_data_upload():
    st.subheader("Bulk Data Upload (Multi-week / Multi-month)")

    st.markdown(
        """
        Use this when you get a **wide** retailer file (not week-by-week uploads).

        Expected format:
        - One sheet per retailer (or retailer name in cell **A1**)
        - Column **A** = SKU (starting row 2)
        - Row **1** from column **B** onward = week ranges (example: `1-1 / 1-3`)
        - Cells = Units sold for that SKU in that week
        - Sales uses your **current pricing** (Vendor Map / Price History). `UnitPrice` is left blank.
        """
    )

    locked_years = load_year_locks()
    years_opt = list(range(this_year - 6, this_year + 2))

    st.markdown("### Year locks")
    cL1, cL2 = st.columns([2, 1])
    with cL1:
        lock_pick = st.multiselect("Locked years (prevent edits)", options=years_opt, default=sorted(list(locked_years)), key="lock_pick")
    with cL2:
        if st.button("Save locks", key="btn_save_locks"):
            save_year_locks(set(int(y) for y in lock_pick))
            st.success("Saved year locks.")
            st.rerun()

    st.divider()

    bulk_upload = st.file_uploader(
        "Upload bulk data workbook (.xlsx)",
        type=["xlsx"],
        key="bulk_up_tab"
    )

    data_year = st.selectbox(
        "Data Year (for header parsing)",
        options=years_opt,
        index=years_opt.index(int(view_year)) if int(view_year) in years_opt else years_opt.index(this_year),
        key="bulk_data_year"
    )

    mode = st.radio(
        "Ingest mode",
        options=["Append (add rows)", "Overwrite year + retailer(s) (replace)"],
        index=0,
        horizontal=True,
        key="bulk_mode"
    )

    is_locked = int(data_year) in load_year_locks()
    if is_locked:
        st.error(f"Year {int(data_year)} is locked. Unlock it above to ingest data for this year.")

    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("Ingest Bulk Workbook", disabled=(bulk_upload is None) or is_locked, key="btn_ingest_bulk"):
            new_rows = read_yow_workbook(bulk_upload, year=int(data_year))

            if mode.startswith("Overwrite"):
                retailers = set(new_rows["Retailer"].dropna().unique().tolist()) if not new_rows.empty else set()
                overwrite_sales_rows(int(data_year), retailers)

            append_sales_to_store(new_rows)
            st.success("Bulk workbook ingested successfully.")
            st.rerun()

    with c2:
        st.caption("Append = adds rows. Overwrite = deletes existing rows for that year + retailer(s) found in the upload, then re-adds.")


def render_seasonality():
        st.subheader("Seasonality (Top 20 seasonal SKUs)")

        if df_all.empty:
            st.info("No sales data yet.")
        else:
            d = df_all.copy()
            d["StartDate"] = pd.to_datetime(d["StartDate"], errors="coerce")
            d = d[d["StartDate"].notna()].copy()
            d["Year"] = d["StartDate"].dt.year.astype(int)
            d["Month"] = d["StartDate"].dt.month.astype(int)
            d["MonthP"] = d["StartDate"].dt.to_period("M")

            month_name = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
            month_list = [month_name[i] for i in range(1,13)]

            years = sorted(d["Year"].unique().tolist())

            c1, c2, c3 = st.columns([1, 2, 2])
            with c1:
                basis = st.radio("Basis", options=["Units", "Sales"], index=0, horizontal=True, key="sea_basis")
            with c2:
                mode = st.selectbox("Timeframe", options=["Pick year", "Lookback"], index=0, key="sea_tf_mode")
            with c3:
                # pick year or lookback window
                if mode == "Pick year":
                    pick_year = st.selectbox("Year", options=["All years"] + [str(y) for y in years], index=0, key="sea_year")
                    month_mode = st.radio("Months", options=["All months (Jan–Dec)", "Custom months"], index=0, horizontal=True, key="sea_month_mode")
                    if month_mode == "Custom months":
                        sel_month_names = st.multiselect("Select months", options=month_list, default=month_list, key="sea_months_pick")
                        sel_months = [k for k,v in month_name.items() if v in sel_month_names]
                    else:
                        sel_months = list(range(1,13))
                    # Apply filters
                    d2 = d[d["Month"].isin(sel_months)].copy()
                    if pick_year != "All years":
                        d2 = d2[d2["Year"] == int(pick_year)].copy()
                else:
                    lookback = st.selectbox("Look back", options=["12 months","24 months","36 months","All available"], index=0, key="sea_lookback")
                    if lookback == "All available":
                        d2 = d.copy()
                    else:
                        n = int(lookback.split()[0])
                        months = sorted(d["MonthP"].dropna().unique().tolist())
                        usem = months[-n:] if len(months) >= n else months
                        d2 = d[d["MonthP"].isin(usem)].copy()

            min_units = st.number_input(
                "Minimum total units (within selected timeframe) to include a SKU",
                min_value=0, max_value=1_000_000, value=20, step=5, key="sea_min_units"
            )

            value_col = "Units" if basis == "Units" else "Sales"

            # Monthly totals per SKU (within timeframe)
            m = d2.groupby(["SKU","MonthP"], as_index=False).agg(v=(value_col,"sum"))

            # Seasonality score computed on month-of-year buckets (Jan..Dec) from the same timeframe
            m_y = d2.groupby(["SKU","Month"], as_index=False).agg(v=(value_col,"sum"))
            tot = m_y.groupby("SKU", as_index=False).agg(total=("v","sum"))
            mx = m_y.sort_values("v", ascending=False).groupby("SKU", as_index=False).first().rename(columns={"Month":"PeakMonth","v":"PeakVal"})
            s = tot.merge(mx, on="SKU", how="left")
            s["SeasonalityScore"] = s["PeakVal"] / s["total"].replace(0, np.nan)

            # Filter by units sold in the timeframe (always units)
            units_tot = d2.groupby("SKU", as_index=False).agg(TotalUnits=("Units","sum"))
            s = s.merge(units_tot, on="SKU", how="left").fillna({"TotalUnits": 0})
            s = s[s["TotalUnits"] >= float(min_units)].copy()

            # Vendor labels
            try:
                if isinstance(vmap, pd.DataFrame) and "SKU" in vmap.columns and "Vendor" in vmap.columns:
                    s = s.merge(vmap[["SKU","Vendor"]].drop_duplicates(), on="SKU", how="left")
            except Exception:
                pass

            s = s.sort_values("SeasonalityScore", ascending=False)
            top = s.head(20).copy()
            top["PeakMonthName"] = top["PeakMonth"].map(month_name)

            st.markdown("### Top seasonal SKUs")
            tbl_cols = ["SKU"]
            if "Vendor" in top.columns:
                tbl_cols.append("Vendor")
            tbl_cols += ["PeakMonthName","SeasonalityScore","TotalUnits"]

            tbl = top[tbl_cols].copy().rename(columns={
                "PeakMonthName": "Peak Month",
                "SeasonalityScore": "Seasonality",
                "TotalUnits": "Total Units",
            })
            tbl = tbl.loc[:, ~tbl.columns.duplicated()].copy()

            st.dataframe(
                tbl.style.format({
                    "Seasonality": lambda v: f"{v*100:.1f}%" if pd.notna(v) else "—",
                    "Total Units": fmt_int,
                }),
                use_container_width=True,
                hide_index=True
            )

            st.divider()
            st.markdown("### Seasonal profiles (monthly totals in timeframe)")

            # Create a complete month index for charting, preserving chronological order
            months_all = sorted(d2["MonthP"].dropna().unique().tolist())
            if not months_all:
                st.info("No months found in the selected timeframe.")
                return

            dt_index = pd.PeriodIndex(months_all, freq="M").to_timestamp()

            for _, row in top.iterrows():
                sku0 = row["SKU"]
                vend0 = row.get("Vendor", "")
                peak0 = row.get("PeakMonthName", "")
                score0 = row.get("SeasonalityScore", np.nan)

                title = f"{sku0}"
                if pd.notna(vend0) and str(vend0).strip():
                    title += f" — {vend0}"
                if pd.notna(score0):
                    title += f" | Peak: {peak0} | Seasonality: {score0*100:.1f}%"

                st.markdown(f"**{title}**")

                prof = m[m["SKU"] == sku0][["MonthP","v"]].copy()
                prof["MonthP"] = prof["MonthP"].astype("period[M]")
                prof = prof.set_index("MonthP").reindex(months_all).fillna(0.0)

                chart_df = pd.DataFrame({f"{basis}": prof["v"].to_numpy()}, index=dt_index)
                st.line_chart(chart_df)

def render_runrate():
        st.subheader("Run-Rate Forecast")

        if df.empty:
            st.info("No sales data yet.")
        else:
            window = st.selectbox("Forecast window (weeks)", options=[4, 8, 12], index=0, key="rr_window")
            lookback = st.selectbox("Lookback for avg", options=[4, 8, 12], index=1, key="rr_lookback")
            level = st.selectbox("Level", options=["SKU", "Vendor", "Retailer"], index=0, key="rr_level")

            d = add_week_col(df)
            weeks = last_n_weeks(d, lookback)
            d = d[d["Week"].isin(weeks)].copy()

            if level == "SKU":
                grp = ["Retailer","Vendor","SKU"]
            elif level == "Vendor":
                grp = ["Vendor"]
            else:
                grp = ["Retailer"]

            base = d.groupby(grp + ["Week"], as_index=False).agg(Units=("Units","sum"), Sales=("Sales","sum"))
            units_piv = base.pivot_table(index=grp, columns="Week", values="Units", aggfunc="sum", fill_value=0.0)
            sales_piv = base.pivot_table(index=grp, columns="Week", values="Sales", aggfunc="sum", fill_value=0.0)

            avg_units = nonzero_mean_rowwise(units_piv).fillna(0.0)
            avg_sales = nonzero_mean_rowwise(sales_piv).fillna(0.0)

            out = avg_units.reset_index().rename(columns={0:"AvgWeeklyUnits"})
            out["AvgWeeklySales"] = avg_sales.values
            out["ProjectedUnits"] = out["AvgWeeklyUnits"] * window
            out["ProjectedSales"] = out["AvgWeeklySales"] * window
            out = out.sort_values("ProjectedSales", ascending=False)

            disp = out.copy()
            disp["AvgWeeklyUnits"] = disp["AvgWeeklyUnits"].round(2)
            disp["ProjectedUnits"] = disp["ProjectedUnits"].round(0).astype(int)

            sty = disp.style.format({
                "AvgWeeklyUnits": lambda v: fmt_2(v),
                "AvgWeeklySales": lambda v: fmt_currency(v),
                "ProjectedUnits": lambda v: fmt_int(v),
                "ProjectedSales": lambda v: fmt_currency(v),
            })
            st.dataframe(sty, use_container_width=True, height=_table_height(disp, max_px=1200), hide_index=True)

            # SKU lookup (only shows when Level includes SKU)
            if isinstance(disp, pd.DataFrame) and (not disp.empty) and ('SKU' in disp.columns):
                st.markdown('---')
                st.markdown('### SKU lookup')
                _sku_list = sorted(disp['SKU'].astype(str).dropna().unique().tolist())
                sel_sku = st.selectbox('Select SKU', options=_sku_list, index=0, key='rr_sku_lookup') if _sku_list else None
                if sel_sku:
                    row_df = disp[disp['SKU'].astype(str) == str(sel_sku)].copy()
                    # keep same formatting
                    row_sty = row_df.style.format({
                        'AvgWeeklyUnits': lambda v: fmt_2(v),
                        'AvgWeeklySales': lambda v: fmt_currency(v),
                        'ProjectedUnits': lambda v: fmt_int(v),
                        'ProjectedSales': lambda v: fmt_currency(v),
                    })
                    st.dataframe(row_sty, use_container_width=True, hide_index=True)


    # -------------------------
    # Seasonality Heatmap
    # -------------------------



def render_alerts():
        st.subheader("Insights & Alerts")

        if df_all.empty:
            st.info("No sales data yet.")
            return

        d = df_all.copy()
        d["StartDate"] = pd.to_datetime(d["StartDate"], errors="coerce")
        d = d[d["StartDate"].notna()].copy()
        d["Year"] = d["StartDate"].dt.year.astype(int)
        d["MonthP"] = d["StartDate"].dt.to_period("M")

        years = sorted(d["Year"].unique().tolist())
        months = sorted(d["MonthP"].unique().tolist())
        month_labels = [m.to_timestamp().strftime("%B %Y") for m in months]
        label_to_period = dict(zip(month_labels, months))

        # --- Period selection ---
        period_mode = st.radio(
            "Period selection",
            options=["Full year (Year vs Year)", "Specific months (A vs B)"],
            index=0,
            horizontal=True,
            key="al_period_mode",
        )

        def _summarize_months(pers: list[pd.Period]) -> str:
            if not pers:
                return "—"
            pers_sorted = sorted(pers)
            labels = [p.to_timestamp().strftime("%b %Y") for p in pers_sorted]
            if len(labels) == 1:
                return labels[0]
            # If they look contiguous, show range; otherwise show count
            try:
                diffs = [(pers_sorted[i+1] - pers_sorted[i]).n for i in range(len(pers_sorted)-1)]
                if diffs and all(int(x) == 1 for x in diffs):
                    return f"{labels[0]}–{labels[-1]}"
            except Exception:
                pass
            return f"{len(labels)} months"

        if period_mode.startswith("Full year"):
            c1, c2 = st.columns(2)
            with c1:
                base_year = st.selectbox("Base Year", options=years, index=0, key="al_base")
            with c2:
                comp_opts = [y for y in years if y != int(base_year)]
                if not comp_opts:
                    st.warning("Only one year of data available. Add another year to compare, or use Specific months.")
                    comp_year = int(base_year)
                else:
                    comp_year = st.selectbox("Comparison Year", options=comp_opts, index=0, key="al_comp")

            a = d[d["Year"] == int(base_year)].copy()
            b = d[d["Year"] == int(comp_year)].copy()
            label_a = str(base_year)
            label_b = str(comp_year)

        else:
            c1, c2 = st.columns(2)
            with c1:
                a_pick = st.multiselect(
                    "Selection A months",
                    options=month_labels,
                    default=month_labels[-1:] if month_labels else [],
                    key="al_a_months",
                )
            with c2:
                b_pick = st.multiselect(
                    "Selection B months",
                    options=month_labels,
                    default=month_labels[-2:-1] if len(month_labels) >= 2 else [],
                    key="al_b_months",
                )

            a_periods = [label_to_period[x] for x in a_pick if x in label_to_period]
            b_periods = [label_to_period[x] for x in b_pick if x in label_to_period]

            # Store for Top SKU movers table
            st.session_state["movers_a_periods"] = [str(p) for p in a_periods]
            st.session_state["movers_b_periods"] = [str(p) for p in b_periods]

            if not a_periods or not b_periods:
                st.info("Pick at least one month in Selection A and Selection B to generate alerts.")
                return

            a = d[d["MonthP"].isin(a_periods)].copy()
            b = d[d["MonthP"].isin(b_periods)].copy()
            label_a = _summarize_months(a_periods)
            label_b = _summarize_months(b_periods)

        basis = st.radio("Basis", options=["Sales", "Units"], index=0, horizontal=True, key="al_basis")
        value_col = "Sales" if basis == "Sales" else "Units"

        insights = []

        # Vendor deltas (worst 5)
        va = a.groupby("Vendor", as_index=False).agg(A=(value_col, "sum"))
        vb = b.groupby("Vendor", as_index=False).agg(B=(value_col, "sum"))
        v = va.merge(vb, on="Vendor", how="outer").fillna(0.0)
        v["Delta"] = v["B"] - v["A"]
        v = v.sort_values("Delta")

        def _fmt(vv):
            return fmt_currency(vv) if value_col == "Sales" else fmt_int(vv)

        for _, row in v.head(5).iterrows():
            if row["Delta"] < 0:
                insights.append(f"🔻 Vendor **{row['Vendor']}** down {_fmt(row['Delta'])} ({label_a} → {label_b}).")

        # Retailer concentration warning (top 1 >= 40%)
        g = b.groupby("Retailer", as_index=False).agg(val=(value_col, "sum")).sort_values("val", ascending=False)
        total = float(g["val"].sum())
        if total > 0 and not g.empty:
            top1_share = float(g.iloc[0]["val"]) / total
            if top1_share >= 0.40:
                insights.append(f"⚠️ Concentration risk: **{g.iloc[0]['Retailer']}** is {top1_share*100:.1f}% of {label_b} ({value_col}).")

        # Growth driven by few SKUs (top10 >= 60% of positive delta)
        sa = a.groupby("SKU", as_index=False).agg(A=(value_col, "sum"))
        sb = b.groupby("SKU", as_index=False).agg(B=(value_col, "sum"))
        sku = sa.merge(sb, on="SKU", how="outer").fillna(0.0)
        sku["Delta"] = sku["B"] - sku["A"]

        pos = sku[sku["Delta"] > 0].sort_values("Delta", ascending=False)
        if not pos.empty:
            top10 = float(pos.head(10)["Delta"].sum())
            total_pos = float(pos["Delta"].sum())
            share = (top10 / total_pos) if total_pos else 0.0
            if share >= 0.60:
                insights.append(f"📈 Growth concentration: top 10 SKUs drive {share*100:.1f}% of positive change ({value_col}) ({label_a} → {label_b}).")

        # Lost SKUs count (had A but not B)
        lost = int(((sku["A"] > 0) & (sku["B"] == 0)).sum())
        if lost:
            insights.append(f"🧯 Lost SKUs: **{lost}** SKUs sold in {label_a} but not in {label_b}.")

        # Year locks notice
        locked = sorted(list(load_year_locks()))
        if locked:
            insights.append(f"🔒 Locked years: {', '.join(str(y) for y in locked)} (bulk ingest blocked).")

        if not insights:
            st.success("No major alerts detected with the current settings.")
        else:
            st.markdown("### Highlights")
            for s in insights:
                st.markdown(f"- {s}")

        with st.expander("Details (tables)", expanded=False):
            st.markdown("**Worst vendors**")
            st.dataframe(
                v.head(15).style.format({"A": _fmt, "B": _fmt, "Delta": _fmt})
                .applymap(lambda v: f"color: {_color(v)};", subset=["Delta"]),
                use_container_width=True,
                hide_index=True
            )

            st.markdown("**Top SKU movers**")
            movers = sku.sort_values("Delta", ascending=False).head(15).copy()
            st.dataframe(
                movers.style.format({"A": _fmt, "B": _fmt, "Delta": _fmt})
                .applymap(lambda v: f"color: {_color(v)};", subset=["Delta"]),
                use_container_width=True,
                hide_index=True
            )


    # Run-Rate Forecast
    # -------------------------


def render_no_sales():
        st.subheader("No Sales SKUs")
        weeks = st.selectbox("Timeframe (weeks)", options=[3,6,8,12], index=0, key="ns_weeks")
        retailers = sorted(vmap["Retailer"].dropna().unique().tolist())
        sel_r = st.selectbox("Retailer", options=["All"] + retailers, index=0, key="ns_retailer")

        if df.empty:
            st.info("No sales data yet.")
        else:
            d2 = df.copy()
            d2["StartDate"] = pd.to_datetime(d2["StartDate"], errors="coerce")
            periods = sorted(d2["StartDate"].dropna().dt.date.unique().tolist())
            use = periods[-weeks:] if len(periods) >= weeks else periods

            if not use:
                st.info("No periods found yet.")
            else:
                sold = d2[d2["StartDate"].dt.date.isin(use)].groupby(["Retailer","SKU"], as_index=False).agg(Units=("Units","sum"), Sales=("Sales","sum"))
                ref = vmap[["Retailer","SKU","Vendor","MapOrder"]].copy()
                if sel_r != "All":
                    ref = ref[ref["Retailer"] == sel_r].copy()

                merged = ref.merge(sold, on=["Retailer","SKU"], how="left")
                merged["Units"] = merged["Units"].fillna(0.0)
                merged["Sales"] = merged["Sales"].fillna(0.0)

                nos = merged[(merged["Units"] <= 0) & (merged["Sales"] <= 0)].copy()
                nos["Status"] = f"No sales in last {weeks} weeks"
                nos = nos.sort_values(["Retailer","MapOrder","SKU"], ascending=[True, True, True])

                out = nos[["Retailer","Vendor","SKU","Status"]].copy()
                st.dataframe(out, use_container_width=True, height=_table_height(out, max_px=1400), hide_index=True)


    # -------------------------
    # WoW Exceptions
    # -------------------------


def keep_total_last(df_in: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """After sorting, keep the TOTAL row (if present) pinned to the bottom."""
    try:
        if df_in is None or df_in.empty or label_col not in df_in.columns:
            return df_in
        m = df_in[label_col].astype(str).str.upper().eq("TOTAL")
        if not m.any():
            return df_in
        total = df_in.loc[m].copy()
        rest = df_in.loc[~m].copy()
        return pd.concat([rest, total], ignore_index=True)
    except Exception:
        return df_in


def resolve_week_dates(periods: list, window):
    """
    periods: sorted list of datetime.date representing week start dates.
    window: int weeks or string like "6 months".
    Returns list of week dates to include, ordered ascending.
    """
    if not periods:
        return []
    if isinstance(window, int):
        return periods[-window:] if len(periods) >= window else periods
    if isinstance(window, str) and "month" in window:
        try:
            n = int(window.split()[0])
        except Exception:
            n = 6
        # get last n unique months present in periods
        months = [pd.Timestamp(d).to_period("M") for d in periods]
        uniq = []
        for p in months:
            if p not in uniq:
                uniq.append(p)
        usem = uniq[-n:] if len(uniq) >= n else uniq
        use = [d for d in periods if pd.Timestamp(d).to_period("M") in usem]
        return use
    return periods


def make_totals_tables(base: pd.DataFrame, group_col: str, tf_weeks, avg_weeks):
    if base.empty:
        return pd.DataFrame(), pd.DataFrame()
    base = base.copy()
    base["StartDate"] = pd.to_datetime(base["StartDate"], errors="coerce")
    periods = sorted(base["StartDate"].dropna().dt.date.unique().tolist())
    first_week = periods[0] if periods else None
    if not periods:
        return pd.DataFrame(), pd.DataFrame()

    use = resolve_week_dates(periods, tf_weeks)
    d = base[base["StartDate"].dt.date.isin(use)].copy()
    d["Week"] = d["StartDate"].dt.date

    sales_p = d.pivot_table(index=group_col, columns="Week", values="Sales", aggfunc="sum", fill_value=0.0).reindex(columns=use, fill_value=0.0)
    units_p = d.pivot_table(index=group_col, columns="Week", values="Units", aggfunc="sum", fill_value=0.0).reindex(columns=use, fill_value=0.0)

    if len(use) >= 2:
        sales_p["Diff"] = sales_p[use[-1]] - sales_p[use[-2]]
        units_p["Diff"] = units_p[use[-1]] - units_p[use[-2]]
    else:
        sales_p["Diff"] = 0.0
        units_p["Diff"] = 0.0

    # Determine which weeks to average based on selected average window
    current_year = int(pd.to_datetime(base["StartDate"], errors="coerce").dt.year.max() or date.today().year)

    # Choose avg weeks from ALL available periods in this filtered dataset (not just the displayed window)
    avg_use = resolve_avg_use(avg_weeks, periods, current_year)

    # Ignore the very first week present (often a partial week)
    if first_week is not None and avg_use:
        avg_use = [w for w in avg_use if pd.to_datetime(w, errors="coerce").date() != first_week]

    # Compute Avg from underlying data so month-year windows can work even if not currently displayed
    if avg_use:
        tmp = base[base["StartDate"].dt.date.isin(avg_use)].copy()
        tmp["Week"] = tmp["StartDate"].dt.date
        s_week = tmp.pivot_table(index=group_col, columns="Week", values="Sales", aggfunc="sum", fill_value=0.0)
        u_week = tmp.pivot_table(index=group_col, columns="Week", values="Units", aggfunc="sum", fill_value=0.0)
        sales_avg = s_week.replace(0, np.nan).mean(axis=1)
        units_avg = u_week.replace(0, np.nan).mean(axis=1)
        sales_p["Avg"] = sales_p.index.to_series().map(sales_avg).fillna(0.0)
        units_p["Avg"] = units_p.index.to_series().map(units_avg).fillna(0.0)
    else:
        sales_p["Avg"] = 0.0
        units_p["Avg"] = 0.0

    # Diff vs Avg uses the last week displayed minus Avg
    if use:
        sales_p["Diff vs Avg"] = sales_p[use[-1]] - sales_p["Avg"]
        units_p["Diff vs Avg"] = units_p[use[-1]] - units_p["Avg"]
    else:
        sales_p["Diff vs Avg"] = 0.0
        units_p["Diff vs Avg"] = 0.0

    sales_p = sales_p.sort_index()
    units_p = units_p.sort_index()

    sales_p.loc["TOTAL"] = sales_p.sum(axis=0)
    units_p.loc["TOTAL"] = units_p.sum(axis=0)

    # Recompute TOTAL Avg and Diff vs Avg from totals row values
    if "Avg" in sales_p.columns and use:
        # Avg already computed; just ensure TOTAL row is numeric
        try:
            sales_p.loc["TOTAL","Avg"] = float(sales_p.loc["TOTAL","Avg"])
            units_p.loc["TOTAL","Avg"] = float(units_p.loc["TOTAL","Avg"])
        except Exception:
            pass
        sales_p.loc["TOTAL","Diff vs Avg"] = sales_p.loc["TOTAL", use[-1]] - sales_p.loc["TOTAL","Avg"]
        units_p.loc["TOTAL","Diff vs Avg"] = units_p.loc["TOTAL", use[-1]] - units_p.loc["TOTAL","Avg"]

    def wlab(c):
        try:
            return pd.Timestamp(c).strftime("%m-%d")
        except Exception:
            return c

    sales_p = sales_p.rename(columns={c: wlab(c) for c in sales_p.columns})
    units_p = units_p.rename(columns={c: wlab(c) for c in units_p.columns})

    return sales_p.reset_index(), units_p.reset_index()

# Retailer Totals

# -------------------------
# Tabs (top navigation)
# -------------------------
# -------------------------
# Main navigation (Product layout)
# -------------------------
(main_overview, main_explorer, main_comparisons, main_data_center, main_admin) = st.tabs([
    "Overview",
    "Sales Explorer",
    "Comparisons",
    "Data Center",
    "Admin",
])

# Sub-tabs (keeps existing render_* functions working without rewriting them)
with main_overview:
    tab_overview, tab_exec = st.tabs(["Overview", "Executive Summary"])

with main_explorer:
    (tab_totals_dash,
     tab_momentum,
     tab_action_center,
     tab_top_skus,
     tab_sku_intel,
     tab_forecasting,
     tab_year_summary,
     tab_alerts) = st.tabs([
        "Totals",
        "Momentum",
        "Action Center",
        "Top SKUs",
        "SKU Intelligence",
        "Forecasting",
        "Year Summary",
        "Alerts",
    ])

with main_comparisons:
    tab_comparisons, tab_wow_exc = st.tabs(["Comparisons", "WoW Exceptions"])

# Data Center + Admin are single panes
tab_data_center = main_data_center
tab_admin = main_admin




# -------------------------
# Momentum + Action Center helpers
# -------------------------

def compute_momentum_scores(df_all: pd.DataFrame, window: int = 8) -> pd.DataFrame:
    """Return SKU momentum scores (0-100) using the last `window` weeks.

    Score blends:
    - Recent growth (last vs first in window)
    - Trend slope (linear fit)
    - Consistency (how often WoW is positive)

    Works off Units and Sales; final score is an average of the two.
    """
    if df_all_raw is None or df_all_raw.empty:
        return pd.DataFrame(columns=["SKU","Momentum","Momentum_Sales","Momentum_Units","Weeks","Lookback Weeks","Up Weeks","Down Weeks","Units_Last","Sales_Last"])

    d = df_all.copy()
    d["EndDate"] = pd.to_datetime(d["EndDate"], errors="coerce")
    d = d[d["EndDate"].notna()].copy()
    if d.empty:
        return pd.DataFrame(columns=["SKU","Momentum","Momentum_Sales","Momentum_Units","Weeks","Lookback Weeks","Up Weeks","Down Weeks","Units_Last","Sales_Last"])

    # Use last N distinct weeks
    weeks = sorted(d["EndDate"].unique())
    weeks = weeks[-window:]
    d = d[d["EndDate"].isin(weeks)].copy()

    sku_week = d.groupby(["SKU","EndDate"], as_index=False)[["Units","Sales"]].sum()

    def _score_series(vals: np.ndarray):
        vals = np.asarray(vals, dtype=float)
        vals = np.nan_to_num(vals, nan=0.0)
        if len(vals) < 2:
            return (0.0, 0.0, 0.0)
        x = np.arange(len(vals), dtype=float)
        # slope
        try:
            slope = np.polyfit(x, vals, 1)[0]
        except Exception:
            slope = 0.0
        # growth
        first = float(vals[0])
        last = float(vals[-1])
        growth = (last - first)
        # consistency: % of positive WoW changes
        diffs = np.diff(vals)
        if len(diffs) == 0:
            pos_rate = 0.0
        else:
            pos_rate = float((diffs > 0).mean())

        # Normalize components within plausible bounds
        # Use robust scaling by median absolute values across series later; here return raw tuple.
        return slope, growth, pos_rate

    rows = []
    for sku, g in sku_week.groupby("SKU"):
        g = g.sort_values("EndDate")
        # Series for momentum window
        s = pd.to_numeric(g['Sales'], errors='coerce').fillna(0.0).to_numpy(dtype=float)
        u = pd.to_numeric(g['Units'], errors='coerce').fillna(0.0).to_numpy(dtype=float)
        up_weeks = int((np.diff(s) > 0).sum()) if len(s) >= 2 else 0
        down_weeks = int((np.diff(s) < 0).sum()) if len(s) >= 2 else 0
        lookback_weeks = int(len(s))

        u = g["Units"].to_numpy(dtype=float)
        s = g["Sales"].to_numpy(dtype=float)
        su = _score_series(u)
        ss = _score_series(s)
        rows.append({
            "SKU": str(sku),
            "Weeks": len(g),
            "Lookback Weeks": int(len(s)),
            "Up Weeks": int((np.diff(s) > 0).sum()) if len(s) >= 2 else 0,
            "Down Weeks": int((np.diff(s) < 0).sum()) if len(s) >= 2 else 0,
            "Lookback Weeks": len(g),
            "Up Weeks": int((np.diff(s) > 0).sum()) if len(s) >= 2 else 0,
            "Down Weeks": int((np.diff(s) < 0).sum()) if len(s) >= 2 else 0,
            "_u_slope": su[0], "_u_growth": su[1], "_u_pos": su[2],
            "_s_slope": ss[0], "_s_growth": ss[1], "_s_pos": ss[2],
            "Units_Last": float(u[-1]) if len(u) else 0.0,
            "Sales_Last": float(s[-1]) if len(s) else 0.0,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["SKU","Momentum","Momentum_Sales","Momentum_Units","Weeks","Lookback Weeks","Up Weeks","Down Weeks","Units_Last","Sales_Last"])

    def _robust_norm(col: str) -> pd.Series:
        v = out[col].astype(float)
        med = float(v.median())
        mad = float((v - med).abs().median())
        if mad == 0:
            return pd.Series(np.zeros(len(v)))
        z = (v - med) / mad
        # squash to 0..1 via logistic
        return 1.0 / (1.0 + np.exp(-z))

    u_slope = _robust_norm("_u_slope")
    u_growth = _robust_norm("_u_growth")
    s_slope = _robust_norm("_s_slope")
    s_growth = _robust_norm("_s_growth")

    # pos_rate already 0..1
    u_pos = out["_u_pos"].clip(0,1)
    s_pos = out["_s_pos"].clip(0,1)

    out["Momentum_Units"] = (0.40*u_growth + 0.35*u_slope + 0.25*u_pos) * 100
    out["Momentum_Sales"] = (0.40*s_growth + 0.35*s_slope + 0.25*s_pos) * 100
    out["Momentum"] = (out["Momentum_Units"] + out["Momentum_Sales"]) / 2.0

    return out[["SKU","Momentum","Momentum_Sales","Momentum_Units","Weeks","Units_Last","Sales_Last"]].sort_values("Momentum", ascending=False)


def forecast_next_weeks(series: pd.Series, periods: int = 4) -> pd.DataFrame:
    """Simple, fast forecast: linear trend + moving average baseline.
    Returns df with columns: t, yhat_trend, yhat_ma, yhat (blend).
    """
    y = pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    n = len(y)
    if n == 0:
        return pd.DataFrame({"t": list(range(1, periods+1)), "yhat": [0.0]*periods})

    x = np.arange(n, dtype=float)
    try:
        slope, intercept = np.polyfit(x, y, 1)
    except Exception:
        slope, intercept = 0.0, float(y.mean()) if n else 0.0

    x_future = np.arange(n, n+periods, dtype=float)
    yhat_trend = intercept + slope * x_future

    # Moving average baseline (last min(4,n) points)
    k = min(4, n)
    ma = float(np.mean(y[-k:]))
    yhat_ma = np.array([ma]*periods, dtype=float)

    # Blend 60% MA + 40% trend (stable)
    yhat = 0.6*yhat_ma + 0.4*yhat_trend

    yhat = np.clip(yhat, 0, None)
    return pd.DataFrame({"t": range(1, periods+1), "yhat_trend": yhat_trend, "yhat_ma": yhat_ma, "yhat": yhat})

# BULLETPROOF_TABS



def compute_momentum_table(df_all: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """
    Rebuilt momentum table:
    - Uses the last `window` weeks (based on EndDate global weeks) to compute:
      SKU | Momentum score (0-100) | Up Weeks | Down Weeks | Units Last | Sales Last
    - Momentum score is a rank-based composite of slope, growth, positive-week count, and current sales.
    """
    if df_all_raw is None or df_all_raw.empty:
        return pd.DataFrame(columns=["SKU","Momentum","Up Weeks","Down Weeks","Units_Last","Sales_Last"])

    df = df_all.copy()
    if "EndDate" not in df.columns or "SKU" not in df.columns:
        return pd.DataFrame(columns=["SKU","Momentum","Up Weeks","Down Weeks","Units_Last","Sales_Last"])

    df["EndDate"] = pd.to_datetime(df["EndDate"], errors="coerce")
    df = df[df["EndDate"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=["SKU","Momentum","Up Weeks","Down Weeks","Units_Last","Sales_Last"])

    # Global last N weeks
    weeks = sorted(df["EndDate"].dropna().unique())
    if not weeks:
        return pd.DataFrame(columns=["SKU","Momentum","Up Weeks","Down Weeks","Units_Last","Sales_Last"])
    window = int(max(2, min(int(window), len(weeks))))
    use_weeks = weeks[-window:]
    sub = df[df["EndDate"].isin(use_weeks)].copy()

    # Aggregate weekly by SKU
    wk = sub.groupby(["SKU","EndDate"], as_index=False)[["Units","Sales"]].sum()
    wk = wk.sort_values(["SKU","EndDate"])

    rows = []
    for sku, g in wk.groupby("SKU", sort=False):
        g = g.sort_values("EndDate")
        s = pd.to_numeric(g["Sales"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        u = pd.to_numeric(g["Units"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if len(s) == 0:
            continue

        diffs = (s[1:] - s[:-1]) if len(s) >= 2 else np.array([])
        up_weeks = int((diffs > 0).sum()) if diffs.size else 0
        down_weeks = int((diffs < 0).sum()) if diffs.size else 0

        # Trend features
        if len(s) >= 2:
            x = np.arange(len(s), dtype=float)
            # slope in $ per week
            try:
                slope = float(np.polyfit(x, s, 1)[0])
            except Exception:
                slope = float(s[-1] - s[0])
            growth = float((s[-1] - s[0]) / (abs(s[0]) + 1e-9))
        else:
            slope = 0.0
            growth = 0.0

        units_last = float(u[-1]) if len(u) else 0.0
        sales_last = float(s[-1]) if len(s) else 0.0

        rows.append({
            "SKU": str(sku),
            "_slope": slope,
            "_growth": growth,
            "_up": up_weeks,
            "_sales_last": sales_last,
            "Up Weeks": up_weeks,
            "Down Weeks": down_weeks,
            "Units_Last": units_last,
            "Sales_Last": sales_last,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["SKU","Momentum","Up Weeks","Down Weeks","Units_Last","Sales_Last"])

    # Rank-based momentum score (0-100) so it is stable and comparable across SKUs each run
    def _pct_rank(series: pd.Series) -> pd.Series:
        try:
            return series.rank(pct=True, method="average").fillna(0.0)
        except Exception:
            return pd.Series([0.0]*len(series), index=series.index)

    slope_p = _pct_rank(out["_slope"])
    growth_p = _pct_rank(out["_growth"])
    up_p = _pct_rank(out["_up"])
    sales_p = _pct_rank(out["_sales_last"])

    score = 40.0 * (0.40*slope_p + 0.30*growth_p + 0.20*up_p + 0.10*sales_p)
    out["Momentum"] = score.round(0).astype(int)

    # Final columns
    final = out[["SKU","Momentum","Up Weeks","Down Weeks","Units_Last","Sales_Last"]].copy()
    final = final.sort_values(["Momentum","Sales_Last"], ascending=[False, False]).reset_index(drop=True)
    return final



def _get_current_and_prev_week(df: pd.DataFrame):
    if df is None or df.empty:
        return None, None, None, None
    d = df.copy()
    d["StartDate"] = pd.to_datetime(d["StartDate"], errors="coerce")
    d["EndDate"] = pd.to_datetime(d["EndDate"], errors="coerce")
    d = d[d["StartDate"].notna() & d["EndDate"].notna()].copy()
    if d.empty:
        return None, None, None, None
    ends = sorted(d["EndDate"].dropna().unique())
    cur_end = ends[-1]
    prev_end = ends[-2] if len(ends) >= 2 else None
    # Use the most common start date for that end date (handles any stray rows)
    cur_start = d.loc[d["EndDate"] == cur_end, "StartDate"].mode()
    cur_start = cur_start.iloc[0] if not cur_start.empty else d["StartDate"].min()
    prev_start = None
    if prev_end is not None:
        prev_start_s = d.loc[d["EndDate"] == prev_end, "StartDate"].mode()
        prev_start = prev_start_s.iloc[0] if not prev_start_s.empty else None
    cur = d[d["EndDate"] == cur_end].copy()
    prev = d[d["EndDate"] == prev_end].copy() if prev_end is not None else d.iloc[0:0].copy()
    return cur, prev, cur_start, cur_end


def _fmt_week_range(start_dt, end_dt) -> str:
    try:
        s = pd.to_datetime(start_dt).date()
        e = pd.to_datetime(end_dt).date()
        return f"{s.strftime('%b %d, %Y')} – {e.strftime('%b %d, %Y')}"
    except Exception:
        return "Current week"


def render_tab_overview():
    with tab_overview:
        st.subheader("Overview")

        if df_all_raw is None or df_all_raw.empty:
            st.info("No sales data yet. Upload your sales_store.csv in Data Management.")
            return

        cur, prev, cur_start, cur_end = _get_current_and_prev_week(df_all_raw)
        if cur is None or cur.empty:
            st.info("No valid weekly rows found (missing StartDate/EndDate).")
            return

        st.caption(f"Week: {_fmt_week_range(cur_start, cur_end)}")

        # What changed? (auto-summary)
        total_sales_cur = float(cur["Sales"].sum())
        total_sales_prev = float(prev["Sales"].sum()) if (prev is not None and not prev.empty) else 0.0
        wow_sales_total = total_sales_cur - total_sales_prev

        # Drivers by Retailer & Vendor (for narrative)
        def _wow_by_key(cur_df, prev_df, key):
            c = cur_df.groupby(key, as_index=False)[["Sales","Units"]].sum()
            p = prev_df.groupby(key, as_index=False)[["Sales","Units"]].sum() if (prev_df is not None and not prev_df.empty) else pd.DataFrame(columns=[key,"Sales","Units"])
            m2 = c.merge(p[[key,"Sales"]].rename(columns={"Sales":"Sales_prev"}), on=key, how="left")
            m2["Sales_prev"] = m2["Sales_prev"].fillna(0.0)
            m2["WoW Sales"] = m2["Sales"] - m2["Sales_prev"]
            # hide entities that have no sales in the selected week (helps prevent 0/0 rows)
            m2 = m2[(pd.to_numeric(m2["Sales"], errors="coerce").fillna(0) > 0) | (pd.to_numeric(m2["Units"], errors="coerce").fillna(0) > 0)].copy()
            return m2.sort_values("WoW Sales")

        wow_r = _wow_by_key(cur, prev, "Retailer") if "Retailer" in cur.columns else pd.DataFrame()
        wow_v = _wow_by_key(cur, prev, "Vendor") if "Vendor" in cur.columns else pd.DataFrame()

        def _money2(v):
            try:
                v = float(v)
            except Exception:
                return "—"
            # Always show 2 decimals and standard negative currency style
            return f"-${abs(v):,.2f}" if v < 0 else f"${v:,.2f}"

        direction = "up" if wow_sales_total > 0 else ("down" if wow_sales_total < 0 else "flat")
        top_r_down = wow_r.head(1) if (wow_r is not None and not wow_r.empty) else pd.DataFrame()
        top_r_up   = wow_r.tail(1) if (wow_r is not None and not wow_r.empty) else pd.DataFrame()
        top_v_down = wow_v.head(1) if (wow_v is not None and not wow_v.empty) else pd.DataFrame()
        top_v_up   = wow_v.tail(1) if (wow_v is not None and not wow_v.empty) else pd.DataFrame()

        with st.expander("🧠 What changed? (auto-summary)", expanded=True):
            st.write(f"WoW sales were **{direction}** {_money2(wow_sales_total)}.")
            if not top_r_down.empty and float(top_r_down["WoW Sales"].iloc[0]) < 0:
                st.write(f"• Biggest retailer headwind: {str(top_r_down['Retailer'].iloc[0])} ({_money2(top_r_down['WoW Sales'].iloc[0])})")
            if not top_v_down.empty and float(top_v_down["WoW Sales"].iloc[0]) < 0:
                st.write(f"• Biggest vendor headwind: {str(top_v_down['Vendor'].iloc[0])} ({_money2(top_v_down['WoW Sales'].iloc[0])})")
            if not top_r_up.empty and float(top_r_up["WoW Sales"].iloc[0]) > 0:
                st.write(f"• Top retailer offset: {str(top_r_up['Retailer'].iloc[0])} ({_money2(top_r_up['WoW Sales'].iloc[0])})")
            if not top_v_up.empty and float(top_v_up["WoW Sales"].iloc[0]) > 0:
                st.write(f"• Top vendor offset: {str(top_v_up['Vendor'].iloc[0])} ({_money2(top_v_up['WoW Sales'].iloc[0])})")


        # Aggregate current + previous
        cur_r = cur.groupby("Retailer", as_index=False)[["Units","Sales"]].sum()
        prev_r = prev.groupby("Retailer", as_index=False)[["Units","Sales"]].sum() if prev is not None and not prev.empty else pd.DataFrame(columns=["Retailer","Units","Sales"])

        cur_v = cur.groupby("Vendor", as_index=False)[["Units","Sales"]].sum()
        prev_v = prev.groupby("Vendor", as_index=False)[["Units","Sales"]].sum() if prev is not None and not prev.empty else pd.DataFrame(columns=["Vendor","Units","Sales"])

        cur_s = cur.groupby("SKU", as_index=False)[["Units","Sales"]].sum()
        prev_s = prev.groupby("SKU", as_index=False)[["Units","Sales"]].sum() if prev is not None and not prev.empty else pd.DataFrame(columns=["SKU","Units","Sales"])

        # Helpers
        def _delta(cur_df, prev_df, key, val_col, key_val):
            try:
                c = float(cur_df.loc[cur_df[key]==key_val, val_col].sum())
            except Exception:
                c = 0.0
            try:
                p = float(prev_df.loc[prev_df[key]==key_val, val_col].sum()) if prev_df is not None and not prev_df.empty else 0.0
            except Exception:
                p = 0.0
            return c, (c - p)

        def _fmt_currency(x):
            """Currency formatter where negatives start with "-" so st.metric colors correctly."""
            try:
                v = float(x)
            except Exception:
                return ""
            return f"-${abs(v):,.2f}" if v < 0 else f"${v:,.2f}"

        def _fmt_int(x):
            try:
                return f"{int(round(float(x))):,}"
            except Exception:
                return ""

        def _posneg_color(v):
            try:
                v = float(v)
            except Exception:
                return ""
            if v > 0:
                return "color: #0f7b0f; font-weight: 600;"
            if v < 0:
                return "color: #b00020; font-weight: 600;"
            return "color: #666;"

        def _top2_cards(label, metric_col):
            is_currency = (metric_col == "Sales")

            top_retailers = cur_r.sort_values([metric_col, ("Units" if metric_col=="Sales" else "Sales")], ascending=False).head(2)
            top_vendors   = cur_v.sort_values([metric_col, ("Units" if metric_col=="Sales" else "Sales")], ascending=False).head(2)
            top_skus      = cur_s.sort_values([metric_col, ("Units" if metric_col=="Sales" else "Sales")], ascending=False).head(2)

            c1, c2, c3 = st.columns(3)

            def _render_two(col, title, key_name, df_top, prev_df):
                col.markdown(f"**{title} ({label})**")
                if df_top.empty:
                    col.caption("—")
                    return
                for i, row in enumerate(df_top.itertuples(index=False), start=1):
                    name = getattr(row, key_name)
                    val, dlt = _delta(df_top, prev_df, key_name, metric_col, name)  # delta helper expects same schema, ok for val
                    # IMPORTANT: delta should be computed from full aggregates, not top slices
                    val, dlt = _delta({"Retailer":cur_r,"Vendor":cur_v,"SKU":cur_s}[key_name],
                                      {"Retailer":prev_r,"Vendor":prev_v,"SKU":prev_s}[key_name],
                                      key_name, metric_col, name)

                    value_str = _fmt_currency(val) if is_currency else _fmt_int(val)
                    delta_str = _fmt_currency(dlt) if is_currency else _fmt_int(dlt)
                    col.metric(f"#{i}: {name}", value_str, delta_str)

            # Render each category with two metrics
            _render_two(c1, "Top Retailers", "Retailer", top_retailers, prev_r)
            _render_two(c2, "Top Vendors", "Vendor", top_vendors, prev_v)
            # Ensure SKU shown as string
            top_skus = top_skus.copy()
            top_skus["SKU"] = top_skus["SKU"].astype(str)
            _render_two(c3, "Top SKUs", "SKU", top_skus, prev_s)

        # Top 2 (Sales) + Top 2 (Units)
        _top2_cards("Sales", "Sales")
        _top2_cards("Units", "Units")

        st.divider()

        # Biggest movers (Top 10 by absolute WoW change)
        movers = cur_s.merge(prev_s, on="SKU", how="left", suffixes=("_cur","_prev"))
        movers["Sales_prev"] = movers["Sales_prev"].fillna(0.0)
        movers["Units_prev"] = movers["Units_prev"].fillna(0.0)
        movers["WoW Sales"] = movers["Sales_cur"] - movers["Sales_prev"]
        movers["WoW Units"] = movers["Units_cur"] - movers["Units_prev"]

        m_sales = movers.assign(_abs=movers["WoW Sales"].abs()).sort_values("_abs", ascending=False).head(10).drop(columns=["_abs"])
        m_units = movers.assign(_abs=movers["WoW Units"].abs()).sort_values("_abs", ascending=False).head(10).drop(columns=["_abs"])

        st.markdown("### Biggest movers this week")
        t1, t2 = st.tabs(["By Sales ($)", "By Units"])

        def _render_movers(df_m):
            if df_m.empty:
                st.info("No movers found (need at least 2 weeks of data).")
                return
            show = df_m[[
                "SKU",
                "Sales_cur","Sales_prev","WoW Sales",
                "Units_cur","Units_prev","WoW Units"
            ]].rename(columns={
                "Sales_cur":"Sales (This Week)",
                "Sales_prev":"Sales (Prev Week)",
                "Units_cur":"Units (This Week)",
                "Units_prev":"Units (Prev Week)",
            }).copy()

            sty = show.style.format({
                "Sales (This Week)": _fmt_currency,
                "Sales (Prev Week)": _fmt_currency,
                "WoW Sales": _fmt_currency,
                "Units (This Week)": _fmt_int,
                "Units (Prev Week)": _fmt_int,
                "WoW Units": _fmt_int,
            }).applymap(_posneg_color, subset=["WoW Sales","WoW Units"])

            st.dataframe(sty, use_container_width=True, hide_index=True)

        with t1:
            _render_movers(m_sales)
        with t2:
            _render_movers(m_units)

        st.divider()

        c4, c5 = st.columns(2)

        # New SKU detected (only show when the SKU has a NEW SALE this week)
        prior = df_all.copy()
        prior["EndDate"] = pd.to_datetime(prior["EndDate"], errors="coerce")
        prior = prior[prior["EndDate"].notna()].copy()
        prior = prior[prior["EndDate"] < cur_end]

        # Treat "sale" as Units > 0 OR Sales > 0. This avoids flagging SKUs that merely appear on a list/map with zero sales.
        cur_sold = cur.copy()
        cur_sold["Units"] = pd.to_numeric(cur_sold.get("Units"), errors="coerce").fillna(0.0)
        cur_sold["Sales"] = pd.to_numeric(cur_sold.get("Sales"), errors="coerce").fillna(0.0)
        cur_sold = cur_sold[(cur_sold["Units"] > 0) | (cur_sold["Sales"] > 0)].copy()

        prior_sold = prior.copy()
        prior_sold["Units"] = pd.to_numeric(prior_sold.get("Units"), errors="coerce").fillna(0.0)
        prior_sold["Sales"] = pd.to_numeric(prior_sold.get("Sales"), errors="coerce").fillna(0.0)
        prior_sold = prior_sold[(prior_sold["Units"] > 0) | (prior_sold["Sales"] > 0)].copy()

        sold_skus_cur = set(cur_sold["SKU"].astype(str))
        sold_skus_prior = set(prior_sold["SKU"].astype(str))
        new_sale_skus = sorted(sold_skus_cur - sold_skus_prior)

        c4.markdown("### New SKU sales detected")
        if not new_sale_skus:
            c4.success("No new-SKU sales this week (nothing sold for the first time).")
        else:
            # Show Units + Sales for the current week for each new-sale SKU
            show = (cur_sold[cur_sold["SKU"].astype(str).isin(new_sale_skus)]
                    .groupby("SKU", as_index=False)[["Units", "Sales"]].sum())

            # Friendly formatting
            show["Units"] = show["Units"].round(0).astype(int)
            show["Sales"] = show["Sales"].apply(_fmt_currency)

            c4.warning(f"{len(new_sale_skus)} SKU(s) recorded their first sale this week.")
            c4.dataframe(show.rename(columns={"SKU": "New Sale SKU"}), hide_index=True, use_container_width=True)

        # Declining alerts
        c5.markdown("### Declining alerts")
        if prev is None or prev.empty:
            c5.info("Need at least 2 weeks of data to calculate declines.")
        else:
            # --- Vendor declines ---
            c5.markdown("**Declining vendor alert**")
            vcmp = cur_v.merge(prev_v, on="Vendor", how="left", suffixes=("_cur","_prev"))
            vcmp["Sales_prev"] = vcmp["Sales_prev"].fillna(0.0)
            vcmp["Units_prev"] = vcmp["Units_prev"].fillna(0.0)
            vcmp["ΔSales"] = vcmp["Sales_cur"] - vcmp["Sales_prev"]
            vcmp["ΔUnits"] = vcmp["Units_cur"] - vcmp["Units_prev"]
            vcmp["%ΔSales"] = np.where(vcmp["Sales_prev"] > 0, vcmp["ΔSales"] / vcmp["Sales_prev"], np.nan)
            vcmp["%ΔUnits"] = np.where(vcmp["Units_prev"] > 0, vcmp["ΔUnits"] / vcmp["Units_prev"], np.nan)

            min_prev_sales = 1000.0
            alerts_v = vcmp[(vcmp["Sales_prev"] >= min_prev_sales) & (vcmp["%ΔSales"] <= -0.25)].copy().sort_values("%ΔSales").head(8)
            if alerts_v.empty:
                c5.success("No major vendor sales declines this week.")
            else:
                c5.warning(f"{len(alerts_v)} vendor(s) down 25%+ WoW (Sales).")
                show_v = alerts_v[["Vendor","Sales_cur","Sales_prev","ΔSales","Units_cur","Units_prev","ΔUnits"]].rename(columns={
                    "Sales_cur":"Sales (This Week)",
                    "Sales_prev":"Sales (Prev Week)",
                    "ΔSales":"WoW Sales",
                    "Units_cur":"Units (This Week)",
                    "Units_prev":"Units (Prev Week)",
                    "ΔUnits":"WoW Units",
                }).copy()
                sty_v = show_v.style.format({
                    "Sales (This Week)": _fmt_currency,
                    "Sales (Prev Week)": _fmt_currency,
                    "WoW Sales": _fmt_currency,
                    "Units (This Week)": _fmt_int,
                    "Units (Prev Week)": _fmt_int,
                    "WoW Units": _fmt_int,
                }).applymap(_posneg_color, subset=["WoW Sales","WoW Units"])
                c5.dataframe(sty_v, hide_index=True, use_container_width=True)

            c5.divider()

            # --- Retailer declines ---
            c5.markdown("**Declining retailer alert**")
            rcmp = cur_r.merge(prev_r, on="Retailer", how="left", suffixes=("_cur","_prev"))
            rcmp["Sales_prev"] = rcmp["Sales_prev"].fillna(0.0)
            rcmp["Units_prev"] = rcmp["Units_prev"].fillna(0.0)
            rcmp["ΔSales"] = rcmp["Sales_cur"] - rcmp["Sales_prev"]
            rcmp["ΔUnits"] = rcmp["Units_cur"] - rcmp["Units_prev"]
            rcmp["%ΔSales"] = np.where(rcmp["Sales_prev"] > 0, rcmp["ΔSales"] / rcmp["Sales_prev"], np.nan)

            min_prev_sales_r = 5000.0  # retailers are usually larger; avoids noise
            alerts_r = rcmp[(rcmp["Sales_prev"] >= min_prev_sales_r) & (rcmp["%ΔSales"] <= -0.25)].copy().sort_values("%ΔSales").head(8)

            if alerts_r.empty:
                c5.success("No major retailer sales declines this week.")
            else:
                c5.warning(f"{len(alerts_r)} retailer(s) down 25%+ WoW (Sales).")
                show_r = alerts_r[["Retailer","Sales_cur","Sales_prev","ΔSales","Units_cur","Units_prev","ΔUnits"]].rename(columns={
                    "Sales_cur":"Sales (This Week)",
                    "Sales_prev":"Sales (Prev Week)",
                    "ΔSales":"WoW Sales",
                    "Units_cur":"Units (This Week)",
                    "Units_prev":"Units (Prev Week)",
                    "ΔUnits":"WoW Units",
                }).copy()
                sty_r = show_r.style.format({
                    "Sales (This Week)": _fmt_currency,
                    "Sales (Prev Week)": _fmt_currency,
                    "WoW Sales": _fmt_currency,
                    "Units (This Week)": _fmt_int,
                    "Units (Prev Week)": _fmt_int,
                    "WoW Units": _fmt_int,
                }).applymap(_posneg_color, subset=["WoW Sales","WoW Units"])
                c5.dataframe(sty_r, hide_index=True, use_container_width=True)




def render_tab_action_center():
    with tab_action_center:
        st.subheader("Action Center (Rebuilt)")
        st.caption("This page is designed to answer: What do we need to DO this week? (Rule-based thresholds)")

        if df_all_raw is None or df_all_raw.empty:
            st.info("No sales data yet.")
            return

        cur, prev, cur_start, cur_end = _get_current_and_prev_week(df_all_raw)
        if cur is None or cur.empty:
            st.info("No valid weekly rows found (missing StartDate/EndDate).")
            return

        # -------- Thresholds (Option 1) --------
        retailer_min_prev = 5000.0
        vendor_min_prev   = 2000.0
        sku_min_prev      = 500.0
        decline_pct       = -0.25   # -25% or worse
        growth_pct        = 0.30    # +30% or better

        new_window_weeks  = 8
        test_max_weeks    = 10
        test_min_pos_wow  = 3

        st.caption(f"Week: {_fmt_week_range(cur_start, cur_end)}")

        # Weekly aggregates
        cur_r = cur.groupby("Retailer", as_index=False)[["Units","Sales"]].sum()
        cur_v = cur.groupby("Vendor",   as_index=False)[["Units","Sales"]].sum()
        cur_s = cur.groupby("SKU",      as_index=False)[["Units","Sales"]].sum()

        prev_r = prev.groupby("Retailer", as_index=False)[["Units","Sales"]].sum() if prev is not None and not prev.empty else pd.DataFrame(columns=["Retailer","Units","Sales"])
        prev_v = prev.groupby("Vendor",   as_index=False)[["Units","Sales"]].sum() if prev is not None and not prev.empty else pd.DataFrame(columns=["Vendor","Units","Sales"])
        prev_s = prev.groupby("SKU",      as_index=False)[["Units","Sales"]].sum() if prev is not None and not prev.empty else pd.DataFrame(columns=["SKU","Units","Sales"])

        def money(x):
            try: return f"${float(x):,.2f}"
            except: return ""
        def pct(x):
            try: return f"{float(x)*100:,.1f}%"
            except: return ""
        def intfmt(x):
            try: return f"{int(round(float(x))):,}"
            except: return ""

        # =======================
        # 1) CRITICAL DECLINES
        # =======================
        st.markdown("## 🔴 Critical Declines")

        colA, colB = st.columns(2)

        # Retailer declines
        with colA:
            st.markdown("### Retailer declines (Sales)")
            if prev_r.empty:
                st.info("Need at least 2 weeks of data to compute WoW declines.")
            else:
                r = cur_r.merge(prev_r, on="Retailer", how="left", suffixes=("_cur","_prev"))
                r["Sales_prev"] = r["Sales_prev"].fillna(0.0)
                r["WoW Sales"]  = r["Sales_cur"] - r["Sales_prev"]
                r["%ΔSales"]    = np.where(r["Sales_prev"] > 0, r["WoW Sales"] / r["Sales_prev"], np.nan)

                r_alert = r[(r["Sales_prev"] >= retailer_min_prev) & (r["%ΔSales"] <= decline_pct)].copy().sort_values("%ΔSales").head(10)

                if r_alert.empty:
                    st.success("No retailer met the decline thresholds this week.")
                else:
                    # SKU drivers per retailer (top 3 ΔSales)
                    cur_rs = cur.groupby(["Retailer","SKU"], as_index=False)[["Sales"]].sum()
                    prev_rs = prev.groupby(["Retailer","SKU"], as_index=False)[["Sales"]].sum() if prev is not None and not prev.empty else pd.DataFrame(columns=["Retailer","SKU","Sales"])
                    rs = cur_rs.merge(prev_rs, on=["Retailer","SKU"], how="left", suffixes=("_cur","_prev"))
                    rs["Sales_prev"] = rs["Sales_prev"].fillna(0.0)
                    rs["ΔSales"] = rs["Sales_cur"] - rs["Sales_prev"]

                    drivers=[]
                    for rr in r_alert["Retailer"].astype(str).tolist():
                        sub = rs[rs["Retailer"].astype(str)==rr].copy().sort_values("ΔSales").head(3)
                        drivers.append(", ".join([f"{str(x.SKU)} ({money(x.ΔSales)})" for x in sub.itertuples(index=False)]))
                    r_alert["Top SKU drivers (ΔSales)"] = drivers

                    show = r_alert[["Retailer","Sales_cur","Sales_prev","WoW Sales","%ΔSales","Top SKU drivers (ΔSales)"]].rename(columns={
                        "Sales_cur":"Sales (This Week)",
                        "Sales_prev":"Sales (Prev Week)",
                    }).copy()
                    show["Sales (This Week)"] = show["Sales (This Week)"].map(money)
                    show["Sales (Prev Week)"] = show["Sales (Prev Week)"].map(money)
                    show["WoW Sales"] = show["WoW Sales"].map(money)
                    show["%ΔSales"] = show["%ΔSales"].map(pct)
                    st.dataframe(show, use_container_width=True, hide_index=True)

        # Vendor declines
        with colB:
            st.markdown("### Vendor declines (Sales)")
            if prev_v.empty:
                st.info("Need at least 2 weeks of data to compute WoW declines.")
            else:
                v = cur_v.merge(prev_v, on="Vendor", how="left", suffixes=("_cur","_prev"))
                v["Sales_prev"] = v["Sales_prev"].fillna(0.0)
                v["WoW Sales"]  = v["Sales_cur"] - v["Sales_prev"]
                v["%ΔSales"]    = np.where(v["Sales_prev"] > 0, v["WoW Sales"] / v["Sales_prev"], np.nan)

                v_alert = v[(v["Sales_prev"] >= vendor_min_prev) & (v["%ΔSales"] <= decline_pct)].copy().sort_values("%ΔSales").head(10)

                if v_alert.empty:
                    st.success("No vendor met the decline thresholds this week.")
                else:
                    # Retailer impacts per vendor (top 3 ΔSales)
                    cur_vr = cur.groupby(["Vendor","Retailer"], as_index=False)[["Sales"]].sum()
                    prev_vr = prev.groupby(["Vendor","Retailer"], as_index=False)[["Sales"]].sum() if prev is not None and not prev.empty else pd.DataFrame(columns=["Vendor","Retailer","Sales"])
                    vr = cur_vr.merge(prev_vr, on=["Vendor","Retailer"], how="left", suffixes=("_cur","_prev"))
                    vr["Sales_prev"] = vr["Sales_prev"].fillna(0.0)
                    vr["ΔSales"] = vr["Sales_cur"] - vr["Sales_prev"]

                    impacts=[]
                    for vv in v_alert["Vendor"].astype(str).tolist():
                        sub = vr[vr["Vendor"].astype(str)==vv].copy().sort_values("ΔSales").head(3)
                        impacts.append(", ".join([f"{str(x.Retailer)} ({money(x.ΔSales)})" for x in sub.itertuples(index=False)]))
                    v_alert["Retailer impacts (ΔSales)"] = impacts

                    show = v_alert[["Vendor","Sales_cur","Sales_prev","WoW Sales","%ΔSales","Retailer impacts (ΔSales)"]].rename(columns={
                        "Sales_cur":"Sales (This Week)",
                        "Sales_prev":"Sales (Prev Week)",
                    }).copy()
                    show["Sales (This Week)"] = show["Sales (This Week)"].map(money)
                    show["Sales (Prev Week)"] = show["Sales (Prev Week)"].map(money)
                    show["WoW Sales"] = show["WoW Sales"].map(money)
                    show["%ΔSales"] = show["%ΔSales"].map(pct)
                    st.dataframe(show, use_container_width=True, hide_index=True)

        # =======================
        # 2) OPPORTUNITIES (FAST GROWERS)
        # =======================
        st.markdown("## 🟢 Opportunities")
        st.markdown("### Fast-growing SKUs (Sales)")

        if prev_s.empty:
            st.info("Need at least 2 weeks of data to compute WoW growth.")
            growers = pd.DataFrame()
        else:
            s = cur_s.merge(prev_s, on="SKU", how="left", suffixes=("_cur","_prev"))
            s["Sales_prev"] = s["Sales_prev"].fillna(0.0)
            s["WoW Sales"]  = s["Sales_cur"] - s["Sales_prev"]
            s["%ΔSales"]    = np.where(s["Sales_prev"] > 0, s["WoW Sales"] / s["Sales_prev"], np.nan)

            growers = s[(s["Sales_prev"] >= sku_min_prev) & (s["%ΔSales"] >= growth_pct)].copy().sort_values("%ΔSales", ascending=False).head(15)

            if growers.empty:
                st.info("No SKUs met the growth thresholds this week.")
            else:
                cur_drv = cur.groupby(["SKU","Retailer","Vendor"], as_index=False)[["Sales"]].sum()
                driver=[]
                for sku in growers["SKU"].astype(str).tolist():
                    sub = cur_drv[cur_drv["SKU"].astype(str)==sku].sort_values("Sales", ascending=False).head(1)
                    driver.append(f"{sub.iloc[0]['Retailer']} / {sub.iloc[0]['Vendor']}" if not sub.empty else "")
                growers["Primary driver"] = driver

                show = growers[["SKU","Primary driver","Sales_cur","Sales_prev","WoW Sales","%ΔSales","Units_cur","Units_prev"]].rename(columns={
                    "Sales_cur":"Sales (This Week)",
                    "Sales_prev":"Sales (Prev Week)",
                    "Units_cur":"Units (This Week)",
                    "Units_prev":"Units (Prev Week)",
                }).copy()
                show["Sales (This Week)"] = show["Sales (This Week)"].map(money)
                show["Sales (Prev Week)"] = show["Sales (Prev Week)"].map(money)
                show["WoW Sales"] = show["WoW Sales"].map(money)
                show["%ΔSales"] = show["%ΔSales"].map(pct)
                show["Units (This Week)"] = show["Units (This Week)"].map(intfmt)
                show["Units (Prev Week)"] = show["Units (Prev Week)"].map(intfmt)
                st.dataframe(show, use_container_width=True, hide_index=True)

        # =======================
        # 3) NEW SKU PERFORMANCE + TEST SIGNALS
        # =======================
        st.markdown("## 🟡 New SKU performance & test signals")

        hist = df_all.copy()
        hist["EndDate"] = pd.to_datetime(hist["EndDate"], errors="coerce")
        hist = hist[hist["EndDate"].notna()].copy()

        cur_end_ts = pd.to_datetime(cur_end)

        sku_first_seen = hist.groupby(hist["SKU"].astype(str))["EndDate"].min()

        # positive-sales rows (used to detect "first sale ever")
        hist_pos = hist[(pd.to_numeric(hist.get("Sales"), errors="coerce").fillna(0) > 0) | (pd.to_numeric(hist.get("Units"), errors="coerce").fillna(0) > 0)].copy()
        sku_first_sales = hist_pos.groupby(hist_pos["SKU"].astype(str))["EndDate"].min() if not hist_pos.empty else pd.Series(dtype="datetime64[ns]")

        # NEW: first-sale by placement (SKU + Retailer)
        if not hist_pos.empty and "Retailer" in hist_pos.columns:
            hist_pos["SKU"] = hist_pos["SKU"].astype(str)
            hist_pos["Retailer"] = hist_pos["Retailer"].astype(str)
            pair_first_sales = hist_pos.groupby(["SKU","Retailer"])["EndDate"].min()
        else:
            pair_first_sales = pd.Series(dtype="datetime64[ns]")

        cur_skus = set(cur["SKU"].astype(str).unique().tolist())

        true_new = []
        activated = []
        for sku in sorted(cur_skus):
            fs = sku_first_seen.get(sku, pd.NaT)
            fz = sku_first_sales.get(sku, pd.NaT)
            if pd.isna(fs):
                continue
            if fs == cur_end_ts:
                true_new.append(sku)
            else:
                # Activated = existed previously, but first positive sale happens this week
                if (not pd.isna(fz)) and (fz == cur_end_ts) and (fs < cur_end_ts):
                    activated.append(sku)

        # NEW: "new somewhere" placements — SKU sold at a Retailer for the first time this week
        new_place_rows = []
        if "Retailer" in cur.columns:
            cur_pos = cur[(pd.to_numeric(cur.get("Sales"), errors="coerce").fillna(0) > 0) | (pd.to_numeric(cur.get("Units"), errors="coerce").fillna(0) > 0)].copy()
            cur_pos["SKU"] = cur_pos["SKU"].astype(str)
            cur_pos["Retailer"] = cur_pos["Retailer"].astype(str)

            # aggregate current week by SKU + Retailer (+ Vendor if present)
            gb_cols = ["SKU","Retailer"] + (["Vendor"] if "Vendor" in cur_pos.columns else [])
            cur_pair = cur_pos.groupby(gb_cols, as_index=False)[["Units","Sales"]].sum()

            for _, r in cur_pair.iterrows():
                sku = str(r["SKU"])
                ret = str(r["Retailer"])
                fpair = pair_first_sales.get((sku, ret), pd.NaT)
                if (not pd.isna(fpair)) and (fpair == cur_end_ts):
                    new_place_rows.append(r.to_dict())

        
        left, right = st.columns(2)
        with left:
            st.markdown("### New SKUs / New Placements this week")

            if not true_new and not activated and not new_place_rows:
                st.info("No true-new SKUs, activated SKUs, or new placements this week.")
            else:
                if true_new:
                    st.success(f"True new (first time ever in data): {len(true_new)}")
                    st.dataframe(pd.DataFrame({"True New SKUs": true_new}), hide_index=True, use_container_width=True)
                if activated:
                    st.warning(f"Activated (first positive sale this week): {len(activated)}")
                    st.dataframe(pd.DataFrame({"Activated SKUs": activated}), hide_index=True, use_container_width=True)

                if new_place_rows:
                    df_np = pd.DataFrame(new_place_rows)
                    # Normalize columns and show where it sold
                    cols = []
                    for c in ["SKU","Vendor","Retailer","Units","Sales"]:
                        if c in df_np.columns:
                            cols.append(c)
                    df_np = df_np[cols].copy()
                    if "Units" in df_np.columns:
                        df_np["Units"] = pd.to_numeric(df_np["Units"], errors="coerce").fillna(0).astype(int)
                    if "Sales" in df_np.columns:
                        df_np["Sales"] = df_np["Sales"].map(money)
                    st.info(f"Sold somewhere NEW for the first time: {len(df_np)}")
                    st.dataframe(df_np, hide_index=True, use_container_width=True)

        # Trending + test passing
        weeks = sorted(hist["EndDate"].unique())
        # For "new window": SKUs whose first_sales is within last `new_window_weeks` weeks
        recent_weeks = weeks[-max(new_window_weeks+2, 12):]
        recent = hist[hist["EndDate"].isin(recent_weeks)].copy()
        sku_week = recent.groupby(["SKU","EndDate"], as_index=False)[["Sales","Units"]].sum().sort_values(["SKU","EndDate"])

        def consec_pos_wow(sku, metric="Sales"):
            g = sku_week[sku_week["SKU"].astype(str)==sku].sort_values("EndDate")
            vals = pd.to_numeric(g[metric], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            if len(vals) < 2:
                return 0
            diffs = np.diff(vals)
            cnt = 0
            for d in diffs[::-1]:
                if d > 0: cnt += 1
                else: break
            return cnt

        trending_rows=[]
        passing_rows=[]
        for sku in sorted(cur_skus):
            f_sales = sku_first_sales.get(sku, pd.NaT)
            f_seen  = sku_first_seen.get(sku, pd.NaT)
            if pd.isna(f_sales):
                continue
            # age in distinct weeks between first_sales and current
            try:
                age_weeks = len([w for w in weeks if w >= f_sales and w <= cur_end_ts])
            except Exception:
                age_weeks = None
            if age_weeks is None:
                continue

            category = "True New" if (not pd.isna(f_seen) and f_seen == f_sales) else "Activated"
            cons = consec_pos_wow(sku, "Sales")

            g2 = hist[(hist["SKU"].astype(str)==sku) & (hist["EndDate"] >= f_sales) & (hist["EndDate"] <= cur_end_ts)].groupby("EndDate", as_index=False)[["Sales","Units"]].sum().sort_values("EndDate")
            sales_vals = pd.to_numeric(g2["Sales"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            pos_wow = int((np.diff(sales_vals) > 0).sum()) if len(sales_vals) >= 2 else 0
            overall_up = (sales_vals[-1] >= sales_vals[0]) if len(sales_vals) >= 2 else True

            cur_tot = cur_s[cur_s["SKU"].astype(str)==sku]
            cur_sales = float(cur_tot["Sales"].sum()) if not cur_tot.empty else 0.0
            cur_units = float(cur_tot["Units"].sum()) if not cur_tot.empty else 0.0

            if age_weeks <= new_window_weeks and cons >= 2 and cur_sales > 0:
                trending_rows.append({
                    "SKU": sku,
                    "Category": category,
                    "Weeks active (sales)": age_weeks,
                    "Consecutive +WoW weeks": cons,
                    "This week sales": cur_sales,
                    "This week units": cur_units,
                })

            if age_weeks <= test_max_weeks and pos_wow >= test_min_pos_wow and overall_up and cur_sales > 0:
                passing_rows.append({
                    "SKU": sku,
                    "Category": category,
                    "Weeks active (sales)": age_weeks,
                    "+WoW weeks (Sales)": pos_wow,
                    "This week sales": cur_sales,
                    "This week units": cur_units,
                })

        
        # Include "new somewhere" placements in trending rules
        if new_place_rows and "Retailer" in hist_pos.columns:
            new_pairs = {(str(r.get("SKU")), str(r.get("Retailer"))) for r in new_place_rows if r.get("SKU") is not None and r.get("Retailer") is not None}

            for (sku, ret) in sorted(new_pairs):
                f_pair = pair_first_sales.get((sku, ret), pd.NaT)
                if pd.isna(f_pair):
                    continue
                try:
                    age_weeks = len([w for w in weeks if w >= f_pair and w <= cur_end_ts])
                except Exception:
                    continue
                if age_weeks is None or age_weeks <= 0:
                    continue
                if age_weeks > new_window_weeks:
                    continue

                sub = hist_pos[(hist_pos["SKU"].astype(str) == sku) & (hist_pos["Retailer"].astype(str) == ret)].copy()
                by_w = sub.groupby("EndDate", as_index=False)[["Sales","Units"]].sum()

                # Align to weeks list
                sales_map = {pd.to_datetime(rw["EndDate"]): float(rw.get("Sales", 0) or 0) for _, rw in by_w.iterrows()}
                units_map = {pd.to_datetime(rw["EndDate"]): float(rw.get("Units", 0) or 0) for _, rw in by_w.iterrows()}
                vals = [sales_map.get(pd.to_datetime(w), 0.0) for w in weeks]
                uvals = [units_map.get(pd.to_datetime(w), 0.0) for w in weeks]

                pos_wow = _consecutive_positive_wow(vals)
                cur_sales = vals[-1] if vals else 0.0
                cur_units = uvals[-1] if uvals else 0.0

                trending_rows.append({
                    "SKU": sku,
                    "Where": ret,
                    "Category": "New Placement",
                    "Weeks active (sales)": age_weeks,
                    "Consecutive +WoW weeks (Sales)": pos_wow,
                    "This week sales": cur_sales,
                    "This week units": cur_units,
                })
        with right:
            st.markdown("### New SKU trending (≤ 8 weeks)")
            if not trending_rows:
                st.info("No new/activated SKUs are trending under the current rules.")
            else:
                df_tr = pd.DataFrame(trending_rows).sort_values(["Weeks active (sales)","Consecutive +WoW weeks (Sales)"], ascending=[True, False])
                if "This week sales" in df_tr.columns:
                    df_tr["This week sales"] = df_tr["This week sales"].map(money)
                if "This week units" in df_tr.columns:
                    df_tr["This week units"] = df_tr["This week units"].map(intfmt)
                st.dataframe(df_tr, hide_index=True, use_container_width=True)

        st.markdown("### ⭐ Test success signals (≤ 10 weeks old, 3+ positive WoW weeks)")
        if not passing_rows:
            st.info("No SKUs are flagged as passing the test rules right now.")
        else:
            df_ps = pd.DataFrame(passing_rows).sort_values(["Weeks active (sales)","+WoW weeks (Sales)"], ascending=[True, False])
            df_ps["This week sales"] = df_ps["This week sales"].map(money)
            df_ps["This week units"] = df_ps["This week units"].map(intfmt)
            st.dataframe(df_ps, hide_index=True, use_container_width=True)

        # =======================
        # 4) ACTION CHECKLIST
        # =======================
        st.markdown("## ✅ Recommended actions")
        actions=[]

        # Use alerts if they exist
        try:
            if not r_alert.empty:
                for rr in r_alert.head(5)["Retailer"].astype(str).tolist():
                    actions.append(f"🔍 Investigate retailer decline: **{rr}** (PO gaps, OOS, program changes).")
        except Exception:
            pass

        try:
            if not v_alert.empty:
                for vv in v_alert.head(5)["Vendor"].astype(str).tolist():
                    actions.append(f"📞 Call vendor: **{vv}** (down WoW — review pricing, replenishment, coverage).")
        except Exception:
            pass

        try:
            if not growers.empty:
                for sku in growers.head(5)["SKU"].astype(str).tolist():
                    actions.append(f"📈 Push expansion / reorder: **SKU {sku}** is growing fast WoW.")
        except Exception:
            pass

        if true_new:
            actions.append(f"🧾 Verify mappings: **{len(true_new)} true-new SKU(s)** (vendor/retailer mapping, pricing).")
        if activated:
            actions.append(f"🧩 Audit activation: **{len(activated)} activated SKU(s)** (why 0 before; test start vs mapping).")
        if passing_rows:
            top_pass = [r['SKU'] for r in passing_rows[:5]]
            actions.append(f"⭐ Consider expansion: passing test signals for **{', '.join(top_pass)}**.")

        if not actions:
            st.success("No urgent actions detected this week based on the current thresholds.")
        else:
            st.markdown("\n".join([f"- {a}" for a in actions]))


def render_tab_momentum():
    with tab_momentum:
        st.subheader("Momentum")
        st.caption("Momentum highlights SKUs with sustained upward sales trends.")

        if df_all_raw is None or df_all_raw.empty:
            st.info("No sales data yet.")
            return

        window = st.slider("Lookback window (weeks)", min_value=4, max_value=52, value=12, step=1)
        mom = compute_momentum_table(df_all, window=window)

        if mom is None or mom.empty:
            st.info("No momentum data available for the selected window.")
            return

        show = mom.copy()

        # SKU lookup (Momentum only)
        sku_q = st.text_input("SKU lookup (Momentum)", value="", placeholder="Type SKU to filter this Momentum table…", key="momentum_sku_lookup")
        if sku_q.strip() and "SKU" in show.columns:
            q = sku_q.strip().lower()
            show = show[show["SKU"].astype(str).str.lower().str.contains(q, na=False)].copy()

        # Ensure numeric for comparisons
        show["Up Weeks"] = pd.to_numeric(show["Up Weeks"], errors="coerce").fillna(0).astype(int)
        show["Down Weeks"] = pd.to_numeric(show["Down Weeks"], errors="coerce").fillna(0).astype(int)
        show["Units_Last"] = pd.to_numeric(show["Units_Last"], errors="coerce").fillna(0).round(0).astype(int)
        show["Sales_Last"] = pd.to_numeric(show["Sales_Last"], errors="coerce").fillna(0.0)

        # Display formatting (match other tables)
        display_df = pd.DataFrame({
            "SKU": show["SKU"].astype(str),
            "Momentum score": show["Momentum"].astype(int),
            "Weeks up": show["Up Weeks"].astype(int),
            "Weeks down": show["Down Weeks"].astype(int),
            "Units last": show["Units_Last"].map(lambda v: f"{int(v):,}"),
            "Sales last": show["Sales_Last"].map(lambda v: f"${float(v):,.2f}"),
        })

        def _highlight_up_down(row):
            up = int(row["Weeks up"])
            down = int(row["Weeks down"])
            styles = [""] * len(row)

            cols = list(row.index)
            up_i = cols.index("Weeks up")
            down_i = cols.index("Weeks down")

            if up > down:
                styles[up_i] = "color: #2ecc71; font-weight: 800;"
            elif down > up:
                styles[down_i] = "color: #e74c3c; font-weight: 800;"
            return styles

        st.dataframe(
            display_df.style.apply(_highlight_up_down, axis=1),
            use_container_width=True,
            hide_index=True
        )



def render_tab_totals_dash():
    with tab_totals_dash:
        st.subheader("Totals Dashboard")

        if df_all.empty:
            st.info("No sales data yet.")
        else:
            d = df_all.copy()
            d["StartDate"] = pd.to_datetime(d["StartDate"], errors="coerce")
            d = d[d["StartDate"].notna()].copy()
            d["Year"] = d["StartDate"].dt.year.astype(int)
            d["Month"] = d["StartDate"].dt.month.astype(int)

            years = sorted(d["Year"].unique().tolist())
            month_name = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
            month_list = [month_name[i] for i in range(1,13)]


            # -------------------------
            # Controls layout (cleaned)
            # -------------------------

            # Row 1: Primary controls
            p1, p2, p3 = st.columns([1, 2, 2])
            with p1:
                year_opt = ["All years"] + [str(y) for y in years]
                pick_year = st.selectbox("Year", options=year_opt, index=0, key="td_year")
            with p2:
                month_mode = st.radio("Months", options=["All months", "Custom months"], index=0, horizontal=True, key="td_month_mode")
                if month_mode == "Custom months":
                    sel_month_names = st.multiselect("Select months", options=month_list, default=month_list, key="td_months")
                    sel_months = [k for k,v in month_name.items() if v in sel_month_names]
                else:
                    sel_months = list(range(1,13))
            with p3:
                tf_opt = st.selectbox(
                    "Weeks shown",
                    options=["4 weeks", "6 weeks", "8 weeks", "13 weeks", "26 weeks", "52 weeks", "3 months", "6 months", "12 months", "All available"],
                    index=1,
                    key="td_tf_weeks"
                )

            d2 = d[d["Month"].isin(sel_months)].copy()
            if pick_year != "All years":
                d2 = d2[d2["Year"] == int(pick_year)].copy()

            # Advanced settings: Group by + Filters + Average window + View
            with st.expander("Advanced settings", expanded=False):
                a0, a1, a2 = st.columns([2, 2, 2])
                with a0:
                    group_by = st.selectbox("Group by", options=["Retailer", "Vendor", "SKU"], index=0, key="td_group_by")
                with a1:
                    month_year_labels = _build_month_year_labels(d2["StartDate"])
                    avg_options = ["4 weeks", "6 weeks", "8 weeks", "13 weeks", "26 weeks", "52 weeks"] + month_year_labels
                    avg_opt = st.selectbox(
                        "Average window",
                        options=avg_options,
                        index=0 if avg_options else 0,
                        key="td_avg_weeks"
                    )
                with a2:
                    view_mode = st.selectbox("View", options=["Weekly (with Diff/Avg)", "Summary totals"], index=0, key="td_view_mode")

                st.markdown("#### Filters")
                f1, f2, f3 = st.columns([2, 2, 2])
                with f1:
                    retailer_filter = st.multiselect(
                        "Retailer filter (optional)",
                        options=sorted([x for x in d2["Retailer"].dropna().unique().tolist() if str(x).strip()]),
                        key="td_retailer_filter"
                    )
                with f2:
                    vendor_filter = st.multiselect(
                        "Vendor filter (optional)",
                        options=sorted([x for x in d2["Vendor"].dropna().unique().tolist() if str(x).strip()]),
                        key="td_vendor_filter"
                    )
                with f3:
                    sku_opts = sorted([x for x in d2["SKU"].dropna().unique().tolist() if str(x).strip()])
                    sku_filter = st.multiselect("SKU filter (optional)", options=sku_opts, key="td_sku_filter")

            # Apply filters
            if retailer_filter:
                d2 = d2[d2["Retailer"].isin(retailer_filter)]
            if vendor_filter:
                d2 = d2[d2["Vendor"].isin(vendor_filter)]
            if sku_filter:
                d2 = d2[d2["SKU"].isin(sku_filter)]


            if d2.empty:
                st.info("No rows match your filters.")
            else:
                        def _tf_map(x):
                            if x == "All available":
                                return "all"
                            if "month" in x:
                                return x
                            try:
                                return int(x.split()[0])
                            except Exception:
                                return 13

                        tf_weeks = _tf_map(tf_opt)

                        if view_mode.startswith("Weekly"):
                            sales_t, units_t = make_totals_tables(d2, group_by, tf_weeks, avg_opt)
                            # Keep alphabetical order for readability
                            if not sales_t.empty and group_by in sales_t.columns:
                                sales_t = sales_t.sort_values(group_by, ascending=True, kind="mergesort")
                                sales_t = keep_total_last(sales_t, group_by)
                            if not units_t.empty and group_by in units_t.columns:
                                units_t = units_t.sort_values(group_by, ascending=True, kind="mergesort")
                                units_t = keep_total_last(units_t, group_by)

                            if sales_t.empty and units_t.empty:
                                st.info("No weekly totals available for the selected filters.")
                            else:
                                tabS, tabU = st.tabs(["Sales", "Units"])

                                with tabS:
                                    _df = sales_t.copy()

                                    def _diff_color(v):
                                        try:
                                            v = float(v)
                                        except Exception:
                                            return ""
                                        if v > 0:
                                            return "color: #2ecc71; font-weight:600;"
                                        if v < 0:
                                            return "color: #e74c3c; font-weight:600;"
                                        return "color: #999999;"

                                    diff_cols = [c for c in _df.columns if c in ["Diff", "Diff vs Avg"]]
                                    sty = _df.style.format({c: fmt_currency for c in _df.columns if c != group_by})
                                    if diff_cols:
                                        sty = sty.applymap(lambda v: _diff_color(v), subset=diff_cols)

                                    # Bold TOTAL row (if present)
                                    try:
                                        if group_by in _df.columns:
                                            total_mask = _df[group_by].astype(str).str.upper().eq("TOTAL")
                                            if total_mask.any():
                                                def _bold_total(row):
                                                    return ["font-weight:700;" if str(row.get(group_by,"")).upper()=="TOTAL" else "" for _ in row]
                                                sty = sty.apply(_bold_total, axis=1)
                                    except Exception:
                                        pass

                                    _max_px = 1600 if group_by == "SKU" else 1200
                                    st.dataframe(
                                        sty,
                                        use_container_width=True,
                                        hide_index=True,
                                        height=_table_height(_df, max_px=_max_px),
                                    )
                                with tabU:
                                    _df = units_t.copy()

                                    def _diff_color(v):
                                        try:
                                            v = float(v)
                                        except Exception:
                                            return ""
                                        if v > 0:
                                            return "color: #2ecc71; font-weight:600;"
                                        if v < 0:
                                            return "color: #e74c3c; font-weight:600;"
                                        return "color: #999999;"

                                    diff_cols = [c for c in _df.columns if c in ["Diff", "Diff vs Avg"]]
                                    sty = _df.style.format({c: fmt_int for c in _df.columns if c != group_by})
                                    if diff_cols:
                                        sty = sty.applymap(lambda v: _diff_color(v), subset=diff_cols)

                                    # Bold TOTAL row (if present)
                                    try:
                                        if group_by in _df.columns:
                                            total_mask = _df[group_by].astype(str).str.upper().eq("TOTAL")
                                            if total_mask.any():
                                                def _bold_total(row):
                                                    return ["font-weight:700;" if str(row.get(group_by,"")).upper()=="TOTAL" else "" for _ in row]
                                                sty = sty.apply(_bold_total, axis=1)
                                    except Exception:
                                        pass

                                    _max_px = 1600 if group_by == "SKU" else 1200
                                    st.dataframe(
                                        sty,
                                        use_container_width=True,
                                        hide_index=True,
                                        height=_table_height(_df, max_px=_max_px),
                                    )
                        else:
                            key = group_by
                            agg = d2.groupby(key, as_index=False).agg(
                                Units=("Units","sum"),
                                Sales=("Sales","sum"),
                                SKUs=("SKU","nunique"),
                            )

                            # Add TOTAL row (always at the bottom) for Retailer/Vendor/SKU views
                            try:
                                total = {
                                    key: "TOTAL",
                                    "Units": float(pd.to_numeric(agg["Units"], errors="coerce").fillna(0).sum()),
                                    "Sales": float(pd.to_numeric(agg["Sales"], errors="coerce").fillna(0).sum()),
                                    "SKUs": float(d2["SKU"].nunique()),
                                }
                                agg = pd.concat([agg, pd.DataFrame([total])], ignore_index=True)
                            except Exception:
                                pass

                            # Alphabetical order + force TOTAL last
                            agg = agg.sort_values(key, ascending=True, kind="mergesort")
                            agg = keep_total_last(agg, key)

                            disp = make_unique_columns(agg)
                            sty = disp.style.format({"Units": fmt_int, "Sales": fmt_currency, "SKUs": fmt_int})
                            # Bold TOTAL row
                            try:
                                if key in disp.columns:
                                    def _bold_total(row):
                                        return ["font-weight:700;" if str(row.get(key,"")).upper()=="TOTAL" else "" for _ in row]
                                    sty = sty.apply(_bold_total, axis=1)
                            except Exception:
                                pass

                            st.dataframe(
                                sty,
                                use_container_width=True,
                                hide_index=True,
                                height=_table_height(disp, max_px=900)
                            )



def render_tab_top_skus():
    with tab_top_skus:
        st.subheader("Top SKUs (across all retailers)")

        if df_all.empty:
            st.info("No sales data yet.")
        else:
            d = df_all.copy()
            d["StartDate"] = pd.to_datetime(d["StartDate"], errors="coerce")
            d = d[d["StartDate"].notna()].copy()
            d["Year"] = d["StartDate"].dt.year.astype(int)
            d["Month"] = d["StartDate"].dt.month.astype(int)

            years = sorted(d["Year"].unique().tolist())
            month_name = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
            month_list = [month_name[i] for i in range(1,13)]

            c1, c2, c3, c4, c5 = st.columns([1, 2, 1, 1, 1])
            with c1:
                year_opt = ["All years"] + [str(y) for y in years]
                pick_year = st.selectbox("Year", options=year_opt, index=0, key="ts_year")
            with c2:
                month_mode = st.radio("Months", options=["All months", "Custom months"], index=0, horizontal=True, key="ts_month_mode")
                if month_mode == "Custom months":
                    sel_month_names = st.multiselect("Select months", options=month_list, default=month_list, key="ts_months")
                    sel_months = [k for k,v in month_name.items() if v in sel_month_names]
                else:
                    sel_months = list(range(1,13))
            with c3:
                sort_by = st.selectbox("Rank by", options=["Sales", "Units"], index=0, key="ts_rank_by")
            with c4:
                top_n = st.number_input("Top N", min_value=10, max_value=5000, value=50, step=10, key="ts_topn")

            with c5:
                min_val = st.number_input(
                    f"Min {sort_by}",
                    min_value=0.0,
                    value=0.0,
                    step=1.0,
                    key="ts_min_val"
                )

            f1, f2 = st.columns([2, 2])
            with f1:
                vendor_filter = st.multiselect(
                    "Vendor filter (optional)",
                    options=sorted([x for x in d["Vendor"].dropna().unique().tolist() if str(x).strip()]),
                    key="ts_vendor_filter"
                )
            with f2:
                retailer_filter = st.multiselect(
                    "Retailer filter (optional)",
                    options=sorted([x for x in d["Retailer"].dropna().unique().tolist() if str(x).strip()]),
                    key="ts_retailer_filter"
                )

            d2 = d[d["Month"].isin(sel_months)].copy()
            if pick_year != "All years":
                d2 = d2[d2["Year"] == int(pick_year)].copy()
            if vendor_filter:
                d2 = d2[d2["Vendor"].isin(vendor_filter)]
            if retailer_filter:
                d2 = d2[d2["Retailer"].isin(retailer_filter)]

            agg = d2.groupby("SKU", as_index=False).agg(
                Units=("Units","sum"),
                Sales=("Sales","sum"),
                Retailers=("Retailer","nunique"),
            )


            # Apply minimum threshold filter (based on Rank by selection)
            if 'min_val' in locals() and min_val and sort_by in agg.columns:
                agg = agg[agg[sort_by].fillna(0) >= float(min_val)].copy()

            if agg.empty:
                st.info("No rows match your filters.")
            else:
                agg = agg.sort_values(sort_by, ascending=False, kind="mergesort").head(int(top_n))
                agg = make_unique_columns(agg)

                st.dataframe(
                    agg.style.format({
                        "Units": fmt_int,
                        "Sales": fmt_currency,
                        "Retailers": fmt_int,
                    }),
                    use_container_width=True,
                    hide_index=True,
                    height=650
                )

                st.divider()
                st.markdown("### SKU lookup (cross-retailer totals + breakdown)")

                sku_q = st.text_input("Type a SKU to inspect (example: EGLAI1)", value="", key="ts_sku_q").strip()
                if sku_q:
                    qn = str(sku_q).strip().upper()
                    dd = d2.copy()
                    dd["SKU_N"] = dd["SKU"].astype(str).str.strip().str.upper()
                    dd = dd[dd["SKU_N"] == qn].copy()

                    if dd.empty:
                        st.warning("No matching rows for that SKU in the current filters.")
                    else:
                        tot_units = float(dd["Units"].sum())
                        tot_sales = float(dd["Sales"].sum())
                        a, b, c = st.columns([1,1,2])
                        a.metric("Total Units", fmt_int(tot_units))
                        b.metric("Total Sales", fmt_currency(tot_sales))
                        c.caption("Breakdown below is by retailer for the selected year/month filters.")

                        by_ret = dd.groupby("Retailer", as_index=False).agg(Units=("Units","sum"), Sales=("Sales","sum"))
                        by_ret = by_ret.sort_values("Sales", ascending=False, kind="mergesort")
                        st.dataframe(
                            by_ret.style.format({"Units": fmt_int, "Sales": fmt_currency}),
                            use_container_width=True,
                            hide_index=True
                        )



def render_tab_wow_exc():
    with tab_wow_exc:
        st.subheader("WoW Exceptions (Most Recent Week vs Prior Average)")

        if df.empty:
            st.info("No sales data yet.")
        else:
            # Use all loaded years for lookbacks, but keep the "end week" anchored to the currently selected view (df)
            d0_all = add_week_col(df_all)
            d0_cur = add_week_col(df) if not df.empty else d0_all.copy()

            weeks_all = sorted(d0_cur["Week"].dropna().unique().tolist())
            if len(weeks_all) < 2:
                st.info("Not enough weeks loaded yet (need at least 2).")
            else:
                scope = st.selectbox("Scope", options=["All", "Retailer", "Vendor"], index=0, key="wow_scope")

                d1_all = d0_all.copy()
                d1_cur = d0_cur.copy()
                if scope == "Retailer":
                    opts = sorted([x for x in d1_all["Retailer"].dropna().unique().tolist() if str(x).strip()])
                    pick = st.selectbox("Retailer", options=opts, index=0 if opts else 0, key="wow_pick_retailer")
                    d1_all = d1_all[d1_all["Retailer"] == pick].copy()
                    d1_cur = d1_cur[d1_cur["Retailer"] == pick].copy()
                elif scope == "Vendor":
                    opts = sorted([x for x in d1_all["Vendor"].dropna().unique().tolist() if str(x).strip()])
                    pick = st.selectbox("Vendor", options=opts, index=0 if opts else 0, key="wow_pick_vendor")
                    d1_all = d1_all[d1_all["Vendor"] == pick].copy()
                    d1_cur = d1_cur[d1_cur["Vendor"] == pick].copy()

                c1, c2, c3 = st.columns([1.2, 1.2, 2.0])
                with c1:
                    # How far back to average (excluding the most recent week)
                    n_prior = st.selectbox(
                        "Prior window",
                        options=["4 weeks", "6 weeks", "8 weeks", "13 weeks", "26 weeks", "52 weeks", "All prior"],
                        index=1,
                        key="wow_prior_window"
                    )
                with c2:
                    basis = st.selectbox("Sort basis", options=["Sales", "Units"], index=0, key="wow_sort_basis")
                with c3:
                    if scope == "All":
                        display_mode = st.radio(
                            "Display mode",
                            options=["SKU totals (all retailers)", "Break out by retailer"],
                            index=0,
                            horizontal=True,
                            key="wow_display_mode"
                        )
                    else:
                        display_mode = "Break out by retailer"

                # Determine most recent week (from current view) + which prior weeks to use (from all history)
                d1_cur = d1_cur[d1_cur["Week"].notna()].copy()
                d1_all = d1_all[d1_all["Week"].notna()].copy()

                weeks_cur = sorted(d1_cur["Week"].dropna().unique().tolist())
                if len(weeks_cur) < 2:
                    st.info("Not enough weeks for this selection.")
                else:
                    end_week = weeks_cur[-1]
                    # prior weeks can extend into previous years
                    prior_weeks_all = sorted([w for w in d1_all["Week"].dropna().unique().tolist() if w < end_week])

                    def _select_prior(prior_weeks):
                        if n_prior == "All prior":
                            return prior_weeks
                        if "month" in str(n_prior).lower():
                            nmo = int(str(n_prior).split()[0])
                            tmp = d1_all[d1_all["Week"].isin(prior_weeks)].copy()
                            tmp["MonthP"] = pd.to_datetime(tmp["StartDate"], errors="coerce").dt.to_period("M")
                            months = sorted(tmp["MonthP"].dropna().unique().tolist())
                            use_months = months[-nmo:] if len(months) >= nmo else months
                            wk = sorted(tmp[tmp["MonthP"].isin(use_months)]["Week"].dropna().unique().tolist())
                            return wk
                        try:
                            n = int(str(n_prior).split()[0])
                        except Exception:
                            n = 6
                        return prior_weeks[-n:] if len(prior_weeks) >= n else prior_weeks

                    prior_weeks = _select_prior(prior_weeks_all)
                    if not prior_weeks:
                        st.info("No prior weeks in the selected window.")
                    else:
                        if display_mode.startswith("SKU totals"):
                            group_cols = ["SKU"]
                            # helpful extra columns
                            extra_aggs = {"Vendor": ("Vendor", lambda s: s.dropna().astype(str).str.strip().iloc[0] if len(s.dropna()) else ""),
                                          "Retailers": ("Retailer", "nunique")}
                        else:
                            group_cols = ["Retailer", "Vendor", "SKU"]
                            extra_aggs = {}

                        dd = d1_all.copy()

                        # Aggregate to weekly grain for each group
                        g = dd.groupby(group_cols + ["Week"], as_index=False).agg(
                            Units=("Units", "sum"),
                            Sales=("Sales", "sum"),
                        )

                        # Split into end week and prior weeks
                        end = g[g["Week"] == end_week].copy()
                        base = g[g["Week"].isin(prior_weeks)].copy()

                        base_avg = base.groupby(group_cols, as_index=False).agg(
                            Units_Base=("Units", "mean"),
                            Sales_Base=("Sales", "mean"),
                        )
                        end_sum = end.groupby(group_cols, as_index=False).agg(
                            Units_End=("Units", "sum"),
                            Sales_End=("Sales", "sum"),
                        )

                        t = end_sum.merge(base_avg, on=group_cols, how="outer").fillna(0.0)
                        t["Units_Diff"] = t["Units_End"] - t["Units_Base"]
                        t["Sales_Diff"] = t["Sales_End"] - t["Sales_Base"]
                        t["Units_% Diff"] = t["Units_Diff"] / t["Units_Base"].replace(0, np.nan)
                        t["Sales_% Diff"] = t["Sales_Diff"] / t["Sales_Base"].replace(0, np.nan)

                        # Add vendor / retailer coverage when in SKU totals mode
                        if display_mode.startswith("SKU totals"):
                            cov = dd.groupby("SKU", as_index=False).agg(
                                Vendor=("Vendor", lambda s: s.dropna().astype(str).str.strip().iloc[0] if len(s.dropna()) else ""),
                                Retailers=("Retailer", "nunique")
                            )
                            t = t.merge(cov, on="SKU", how="left")

                        sort_col = "Sales_Diff" if basis == "Sales" else "Units_Diff"
                        t = t.sort_values(sort_col, ascending=True, kind="mergesort")  # show biggest negatives first

                        # Keep useful column order
                        if display_mode.startswith("SKU totals"):
                            cols = ["SKU", "Vendor", "Retailers",
                                    "Units_Base", "Units_End", "Units_Diff", "Units_% Diff",
                                    "Sales_Base", "Sales_End", "Sales_Diff", "Sales_% Diff"]
                        else:
                            cols = ["Retailer", "Vendor", "SKU",
                                    "Units_Base", "Units_End", "Units_Diff", "Units_% Diff",
                                    "Sales_Base", "Sales_End", "Sales_Diff", "Sales_% Diff"]
                        cols = [c for c in cols if c in t.columns]
                        t = t[cols].copy()

                        # Totals row at bottom for quick reference
                        try:
                            total = {c: "" for c in t.columns}
                            first = t.columns[0]
                            total[first] = "TOTAL"
                            for c in t.columns:
                                if c in {"SKU","Vendor","Retailer"}:
                                    continue
                                if c == "Retailers":
                                    total[c] = float(dd["Retailer"].nunique())
                                else:
                                    total[c] = float(pd.to_numeric(t[c], errors="coerce").fillna(0).sum())
                            t = pd.concat([t, pd.DataFrame([total])], ignore_index=True)
                        except Exception:
                            pass

                        # Styling
                        disp = make_unique_columns(t)

                        def _diff_color(v):
                            try:
                                v = float(v)
                            except Exception:
                                return ""
                            if v > 0:
                                return "color: #2ecc71; font-weight:600;"
                            if v < 0:
                                return "color: #e74c3c; font-weight:600;"
                            return "color: #999999;"

                        sty = disp.style.format({
                            "Units_Base": fmt_int,
                            "Units_End": fmt_int,
                            "Units_Diff": fmt_int_signed,
                            "Units_% Diff": lambda v: f"{(v*100):.1f}%" if pd.notna(v) else "—",
                            "Sales_Base": fmt_currency,
                            "Sales_End": fmt_currency,
                            "Sales_Diff": fmt_currency_signed,
                            "Sales_% Diff": lambda v: f"{(v*100):.1f}%" if pd.notna(v) else "—",
                            "Retailers": fmt_int,
                        })

                        for c in ["Units_Diff", "Sales_Diff"]:
                            if c in disp.columns:
                                sty = sty.applymap(lambda v: _diff_color(v), subset=[c])

                        # Bold TOTAL row (if present)
                        try:
                            first = disp.columns[0]
                            if first in disp.columns:
                                def _bold_total(row):
                                    return ["font-weight:700;" if str(row.get(first,"")).upper()=="TOTAL" else "" for _ in row]
                                sty = sty.apply(_bold_total, axis=1)
                        except Exception:
                            pass

                        st.caption(
                            f"Comparing most recent week ({pd.Timestamp(end_week).strftime('%m-%d')}) "
                            f"to the average of prior {len(prior_weeks)} week(s): "
                            + ", ".join([pd.Timestamp(w).strftime('%m-%d') for w in prior_weeks])
                        )
                        st.dataframe(sty, use_container_width=True, height=_table_height(disp, max_px=1200), hide_index=True)

                        # SKU lookup (based on this WoW Exceptions table + current filters)
                        if isinstance(disp, pd.DataFrame) and (not disp.empty) and ('SKU' in disp.columns):
                            st.markdown('---')
                            st.markdown('### SKU lookup')
                            _sku_list = sorted(disp['SKU'].astype(str).dropna().unique().tolist())
                            sel_sku = st.selectbox('Select SKU', options=_sku_list, index=0, key='wow_sku_lookup') if _sku_list else None
                            if sel_sku:
                                row_df = disp[disp['SKU'].astype(str) == str(sel_sku)].copy()
                                st.dataframe(row_df, use_container_width=True, hide_index=True)



def render_tab_exec():
    with tab_exec:
        st.subheader("Executive Summary")


        scope = st.selectbox("Scope", options=["All", "Retailer", "Vendor"], index=0, key="ex_scope")

        # Filter scope
        if scope == "Retailer":
            opts = sorted([x for x in df_all["Retailer"].dropna().unique().tolist() if str(x).strip()])
            pick = st.selectbox("Retailer", options=opts, index=0 if opts else 0, key="ex_pick_r")
            d = df_all[df_all["Retailer"] == pick].copy()
            title = f"Executive Summary - {pick}"
        elif scope == "Vendor":
            opts = sorted([x for x in df_all["Vendor"].dropna().unique().tolist() if str(x).strip()])
            pick = st.selectbox("Vendor", options=opts, index=0 if opts else 0, key="ex_pick_v")
            d = df_all[df_all["Vendor"] == pick].copy()
            title = f"Executive Summary - {pick}"
        else:
            d = df_all.copy()
            title = "Executive Summary - All Retailers"

        d["StartDate"] = pd.to_datetime(d["StartDate"], errors="coerce")
        d = d[d["StartDate"].notna()].copy()
        if d.empty:
            st.info("No rows in the selected scope/year.")
            st.stop()

        d["Year"] = d["StartDate"].dt.year.astype(int)
        years = sorted(d["Year"].unique().tolist())

        # Keep full-scope history for the multi-year table at the bottom
        d_scope_all = d.copy()

        pick_year = st.selectbox("Year", options=years, index=(len(years)-1 if years else 0), key="ex_year_pick")
        d = d_scope_all[d_scope_all["Year"] == int(pick_year)].copy()

        st.caption(title)

        # KPI row
        m = wow_mom_metrics(d)
        cols = st.columns(6)
        cols[0].metric("Units", fmt_int(m["total_units"]))
        cols[1].metric("Sales", fmt_currency(m["total_sales"]))
        cols[2].markdown(f"<div style='color:{_color(m['wow_units'])}; font-weight:600;'>WoW Units: {fmt_int(m['wow_units']) if m['wow_units'] is not None else '—'}</div>", unsafe_allow_html=True)
        cols[3].markdown(f"<div style='color:{_color(m['wow_sales'])}; font-weight:600;'>WoW Sales: {fmt_currency(m['wow_sales']) if m['wow_sales'] is not None else '—'}</div>", unsafe_allow_html=True)
        cols[4].markdown(f"<div style='color:{_color(m['mom_units'])}; font-weight:600;'>MoM Units: {fmt_int(m['mom_units']) if m['mom_units'] is not None else '—'}</div>", unsafe_allow_html=True)
        cols[5].markdown(f"<div style='color:{_color(m['mom_sales'])}; font-weight:600;'>MoM Sales: {fmt_currency(m['mom_sales']) if m['mom_sales'] is not None else '—'}</div>", unsafe_allow_html=True)

        st.divider()

        # When Scope = ALL: show one line per SKU (combined across all retailers)
        if scope == "All":
            sku = d.groupby("SKU", as_index=False).agg(
                Vendor=("Vendor", lambda s: (s.dropna().astype(str).str.strip().replace("", np.nan).dropna().iloc[0] if s.dropna().astype(str).str.strip().replace("", np.nan).dropna().shape[0] else "Unmapped")),
                Retailers=("Retailer", "nunique"),
                TotalUnits=("Units", "sum"),
                TotalSales=("Sales", "sum"),
            )
            sku = sku.sort_values("TotalSales", ascending=False, kind="mergesort")

            disp = sku[["SKU","Vendor","Retailers","TotalUnits","TotalSales"]].copy()
            st.markdown("### SKU totals (all retailers combined)")
            st.dataframe(
                disp.style.format({"Retailers": fmt_int, "TotalUnits": fmt_int, "TotalSales": fmt_currency}),
                use_container_width=True,
                hide_index=True,
                height=_table_height(disp, max_px=1100),
            )

        else:
            # Monthly totals table (keep as-is)
            d2 = d.copy()
            d2["MonthP"] = d2["StartDate"].dt.to_period("M")
            mon = d2.groupby("MonthP", as_index=False).agg(Units=("Units","sum"), Sales=("Sales","sum")).sort_values("MonthP")
            if not mon.empty:
                mon["Month"] = mon["MonthP"].map(month_label)
                mon = mon[["Month","Units","Sales"]]
                st.markdown("### Monthly totals")
                st.dataframe(
                    mon.style.format({"Units": fmt_int, "Sales": fmt_currency}),
                    use_container_width=True,
                    height=_table_height(mon, max_px=800),
                    hide_index=True
                )

            # Mix table (keep as-is)
            if scope == "Retailer":
                mix = d.groupby("Vendor", as_index=False).agg(Units=("Units","sum"), Sales=("Sales","sum"))
                mix = mix[(mix["Units"].fillna(0) > 0) | (mix["Sales"].fillna(0) > 0)]
                total_u = float(mix["Units"].sum()) if not mix.empty else 0.0
                total_s = float(mix["Sales"].sum()) if not mix.empty else 0.0
                mix["% Units"] = mix["Units"].apply(lambda v: (v/total_u) if total_u else 0.0)
                mix["% Sales"] = mix["Sales"].apply(lambda v: (v/total_s) if total_s else 0.0)
                mix = mix.sort_values("% Sales", ascending=False, kind="mergesort")
                st.markdown("### Vendor mix")
            else:
                mix = d.groupby("Retailer", as_index=False).agg(Units=("Units","sum"), Sales=("Sales","sum"))
                mix = mix[(mix["Units"].fillna(0) > 0) | (mix["Sales"].fillna(0) > 0)]
                total_u = float(mix["Units"].sum()) if not mix.empty else 0.0
                total_s = float(mix["Sales"].sum()) if not mix.empty else 0.0
                mix["% Units"] = mix["Units"].apply(lambda v: (v/total_u) if total_u else 0.0)
                mix["% Sales"] = mix["Sales"].apply(lambda v: (v/total_s) if total_s else 0.0)
                mix = mix.sort_values("% Sales", ascending=False, kind="mergesort")
                st.markdown("### Retailer mix")

            st.dataframe(
                mix.style.format({"Units": fmt_int, "Sales": fmt_currency, "% Units": lambda v: f"{v*100:.1f}%", "% Sales": lambda v: f"{v*100:.1f}%"}),
                use_container_width=True,
                height=_table_height(mix, max_px=900),
                hide_index=True
            )

            st.divider()

            # Top / Bottom SKUs (keep the same idea as before)
            sold = d.groupby(["SKU","Vendor"], as_index=False).agg(Units=("Units","sum"), Sales=("Sales","sum"))
            sold = sold[(sold["Units"].fillna(0) > 0) | (sold["Sales"].fillna(0) > 0)].copy()

            left, right = st.columns(2)
            with left:
                st.markdown("### Top 10 SKUs (by Sales)")
                top10 = sold.sort_values("Sales", ascending=False, kind="mergesort").head(10)[["SKU","Vendor","Units","Sales"]]
                st.dataframe(top10.style.format({"Units": fmt_int, "Sales": fmt_currency}), use_container_width=True, hide_index=True, height=_table_height(top10, max_px=520))
            with right:
                st.markdown("### Bottom 10 SKUs (by Sales)")
                bot10 = sold.sort_values("Sales", ascending=True, kind="mergesort").head(10)[["SKU","Vendor","Units","Sales"]]
                st.dataframe(bot10.style.format({"Units": fmt_int, "Sales": fmt_currency}), use_container_width=True, hide_index=True, height=_table_height(bot10, max_px=520))
            st.divider()
            st.markdown("### Multi-year totals (always shown)")
            try:
                d_all_scope = d_scope_all.copy()
                d_all_scope["StartDate"] = pd.to_datetime(d_all_scope["StartDate"], errors="coerce")
                d_all_scope = d_all_scope[d_all_scope["StartDate"].notna()].copy()
                d_all_scope["Year"] = d_all_scope["StartDate"].dt.year.astype(int)

                # Choose dimension for the table:
                if scope == "Retailer":
                    dim = "Vendor"
                    label = "Vendors"
                elif scope == "Vendor":
                    dim = "Retailer"
                    label = "Retailers"
                else:
                    dim = "Retailer"
                    label = "Retailers"

                years_all = sorted(d_all_scope["Year"].unique().tolist())
                if years_all:
                    years_show = years_all[-4:] if len(years_all) > 4 else years_all  # keep recent 4 by default
                else:
                    years_show = []

                # Aggregate and pivot
                g = d_all_scope.groupby([dim, "Year"], as_index=False).agg(Units=("Units","sum"), Sales=("Sales","sum"))
                u = g.pivot_table(index=dim, columns="Year", values="Units", aggfunc="sum", fill_value=0.0)
                s = g.pivot_table(index=dim, columns="Year", values="Sales", aggfunc="sum", fill_value=0.0)

                # Build a combined display with Units_YYYY and Sales_YYYY columns
                out = pd.DataFrame({dim: u.index}).reset_index(drop=True)
                for y in years_all:
                    out[f"Units {y}"] = u.get(y, 0.0).values if y in u.columns else 0.0
                    out[f"Sales {y}"] = s.get(y, 0.0).values if y in s.columns else 0.0

                # Totals row
                try:
                    total = {dim: "TOTAL"}
                    for c in out.columns:
                        if c == dim:
                            continue
                        total[c] = float(pd.to_numeric(out[c], errors="coerce").fillna(0).sum())
                    out = pd.concat([out, pd.DataFrame([total])], ignore_index=True)
                except Exception:
                    pass

                # Sort by latest Sales year (excluding TOTAL)
                try:
                    latest_year = max(years_all) if years_all else None
                    if latest_year is not None and f"Sales {latest_year}" in out.columns:
                        m_total = out[dim].astype(str).str.upper().eq("TOTAL")
                        rest = out.loc[~m_total].sort_values(f"Sales {latest_year}", ascending=False, kind="mergesort")
                        out = pd.concat([rest, out.loc[m_total]], ignore_index=True)
                except Exception:
                    pass

                fmt = {}
                for c in out.columns:
                    if c.startswith("Units "):
                        fmt[c] = fmt_int
                    if c.startswith("Sales "):
                        fmt[c] = fmt_currency

                st.caption(f"{label} across years (independent of the Year dropdown above).")
                st.dataframe(out.style.format(fmt), use_container_width=True, hide_index=True, height=_table_height(out, max_px=1100))
            except Exception:
                st.caption("Multi-year table will appear when historical data is available.")

        st.markdown("---")
        st.markdown("### Executive Summary PDF Export")
        st.caption("Exports will use exactly what you have selected above (Scope + selection + current View Year).")

        # Persist the current Executive Summary dataset for other tabs/exports
        try:
            st.session_state["exec_export_scope"] = scope
            st.session_state["exec_export_title"] = title
            st.session_state["exec_export_df"] = d.copy()
        except Exception:
            pass

        if "exec_onepager_pdf_bytes" not in st.session_state:
            st.session_state.exec_onepager_pdf_bytes = None
            st.session_state.exec_onepager_pdf_name = None

        colA, colB = st.columns([1,1])
        with colA:
            build_btn = st.button("Build Executive One‑Pager (PDF)", use_container_width=True, key="ex_onepager_build_btn")
        with colB:
            include_table = st.checkbox("Include Top SKUs table", value=True, key="ex_onepager_include_table")

        if build_btn:
            try:
                df_src = d.copy()
                # Current vs previous week for deltas (within the same filtered dataset)
                cur, prev, cur_start, cur_end = _get_current_and_prev_week(df_src)
                sales_cur = float(cur["Sales"].sum()) if (cur is not None and not cur.empty and "Sales" in cur.columns) else float(df_src["Sales"].sum() if "Sales" in df_src.columns else 0.0)
                units_cur = float(cur["Units"].sum()) if (cur is not None and not cur.empty and "Units" in cur.columns) else float(df_src["Units"].sum() if "Units" in df_src.columns else 0.0)

                sales_prev = float(prev["Sales"].sum()) if (prev is not None and not prev.empty and "Sales" in prev.columns) else 0.0
                units_prev = float(prev["Units"].sum()) if (prev is not None and not prev.empty and "Units" in prev.columns) else 0.0

                wow_sales = sales_cur - sales_prev
                wow_units = units_cur - units_prev
                wow_sales_pct = (wow_sales / sales_prev) if sales_prev else None
                wow_units_pct = (wow_units / units_prev) if units_prev else None

                active_skus = int(df_src["SKU"].astype(str).nunique()) if "SKU" in df_src.columns else 0
                vendors_n = int(df_src["Vendor"].astype(str).nunique()) if "Vendor" in df_src.columns else 0
                retailers_n = int(df_src["Retailer"].astype(str).nunique()) if "Retailer" in df_src.columns else 0

                subtitle = title
                if cur is not None and not cur.empty and pd.notna(cur_start) and pd.notna(cur_end):
                    subtitle = f"{title} • Week {pd.to_datetime(cur_start).date()} to {pd.to_datetime(cur_end).date()}"

                def _money_delta(v, pct=None):
                    s = _fmt_pdf_money(v)
                    if pct is None:
                        return s if s != "" else None
                    try:
                        return f"{s} ({pct*100:+.1f}%)"
                    except Exception:
                        return s

                kpis = [
                    ("Sales", _fmt_pdf_money(sales_cur), _money_delta(wow_sales, wow_sales_pct)),
                    ("Units", _fmt_pdf_int(units_cur), (f"{_fmt_pdf_int(wow_units)} ({wow_units_pct*100:+.1f}%)" if wow_units_pct is not None else _fmt_pdf_int(wow_units))),
                    ("Active SKUs", f"{active_skus:,}", None),
                    ("Vendors", f"{vendors_n:,}", None),
                    ("Retailers", f"{retailers_n:,}", None),
                ]

                bullets = []
                if scope == "Vendor":
                    bullets.append(f"Vendor focus: {st.session_state.get('ex_pick_v', '')}".strip())
                elif scope == "Retailer":
                    bullets.append(f"Retailer focus: {st.session_state.get('ex_pick_r', '')}".strip())

                bullets.append(f"Total sales: {_fmt_pdf_money(sales_cur)} on {_fmt_pdf_int(units_cur)} units.")
                if cur is not None and not cur.empty:
                    bullets.append(f"WoW change: {_fmt_pdf_money(wow_sales)} sales, {_fmt_pdf_int(wow_units)} units.")
                if "SKU" in df_src.columns:
                    top_sku = df_src.groupby("SKU", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False).head(1)
                    if not top_sku.empty:
                        bullets.append(f"Top SKU by sales: {str(top_sku['SKU'].iloc[0])} ({_fmt_pdf_money(top_sku['Sales'].iloc[0])}).")
                if "Retailer" in df_src.columns:
                    top_r = df_src.groupby("Retailer", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False).head(1)
                    if not top_r.empty:
                        bullets.append(f"Top retailer by sales: {str(top_r['Retailer'].iloc[0])} ({_fmt_pdf_money(top_r['Sales'].iloc[0])}).")

                table_df = None
                if include_table:
                    try:
                        sold_pdf = df_src.groupby(["SKU","Vendor"], as_index=False).agg(Units=("Units","sum"), Sales=("Sales","sum"))
                        sold_pdf = sold_pdf.sort_values("Sales", ascending=False, kind="mergesort").head(10).copy()
                        sold_pdf["Units"] = sold_pdf["Units"].map(_fmt_pdf_int)
                        sold_pdf["Sales"] = sold_pdf["Sales"].map(_fmt_pdf_money)
                        table_df = sold_pdf[["SKU","Vendor","Units","Sales"]]
                    except Exception:
                        table_df = None

                pdf_bytes = make_one_pager_pdf("Cornerstone Sales Dashboard — Executive One‑Pager", subtitle, kpis, [b for b in bullets if b], table_df)
                st.session_state.exec_onepager_pdf_bytes = pdf_bytes
                safe_name = re.sub(r"[^A-Za-z0-9_\-]+", "_", title).strip("_")
                st.session_state.exec_onepager_pdf_name = f"Executive_OnePager_{safe_name}_{int(view_year)}.pdf"
                st.success("Executive one‑pager is ready below.")
            except Exception as e:
                st.error(f"One‑pager build failed: {e}")

        if st.session_state.exec_onepager_pdf_bytes:
            st.download_button(
                "⬇️ Download Executive One‑Pager (PDF)",
                data=st.session_state.exec_onepager_pdf_bytes,
                file_name=st.session_state.exec_onepager_pdf_name or "Executive_OnePager.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="ex_onepager_dl_btn",
            )

        st.divider()
        st.markdown("### Weekly Summary Export")
        st.caption("One-click export of the current week highlights (Action Center) to Excel and PDF.")

        if d is not None and not d.empty:
            cur, prev, cur_start, cur_end = _get_current_and_prev_week(d)
            if cur is not None and not cur.empty:
                # Build the same core tables used in Action Center
                cur_r = cur.groupby("Retailer", as_index=False)[["Units","Sales"]].sum()
                prev_r = prev.groupby("Retailer", as_index=False)[["Units","Sales"]].sum() if prev is not None and not prev.empty else pd.DataFrame(columns=["Retailer","Units","Sales"])
                cur_v = cur.groupby("Vendor", as_index=False)[["Units","Sales"]].sum()
                prev_v = prev.groupby("Vendor", as_index=False)[["Units","Sales"]].sum() if prev is not None and not prev.empty else pd.DataFrame(columns=["Vendor","Units","Sales"])
                cur_s = cur.groupby("SKU", as_index=False)[["Units","Sales"]].sum()
                prev_s = prev.groupby("SKU", as_index=False)[["Units","Sales"]].sum() if prev is not None and not prev.empty else pd.DataFrame(columns=["SKU","Units","Sales"])

                movers = cur_s.merge(prev_s, on="SKU", how="left", suffixes=("_cur","_prev"))
                movers["Sales_prev"] = movers["Sales_prev"].fillna(0.0)
                movers["Units_prev"] = movers["Units_prev"].fillna(0.0)
                movers["WoW Sales"] = movers["Sales_cur"] - movers["Sales_prev"]
                movers["WoW Units"] = movers["Units_cur"] - movers["Units_prev"]
                movers["_abs_wow_sales"] = movers["WoW Sales"].abs()
                movers = movers.sort_values("_abs_wow_sales", ascending=False).head(25).drop(columns=["_abs_wow_sales"])
                prior = d.copy()
                prior["EndDate"] = pd.to_datetime(prior["EndDate"], errors="coerce")
                prior = prior[prior["EndDate"].notna()].copy()
                prior = prior[prior["EndDate"] < cur_end]
                new_skus = sorted(set(cur["SKU"].astype(str)) - set(prior["SKU"].astype(str)))

                mom = compute_momentum_scores(d, window=8)

                def _declines(cur_df, prev_df, key, min_prev_sales=1000.0, pct=-0.25):
                    if prev_df is None or prev_df.empty:
                        return pd.DataFrame()
                    cmp = cur_df.merge(prev_df, on=key, how="left", suffixes=("_cur","_prev"))
                    cmp["Sales_prev"] = cmp["Sales_prev"].fillna(0.0)
                    cmp["Units_prev"] = cmp["Units_prev"].fillna(0.0)
                    cmp["WoW Sales"] = cmp["Sales_cur"] - cmp["Sales_prev"]
                    cmp["WoW Units"] = cmp["Units_cur"] - cmp["Units_prev"]
                    cmp["%ΔSales"] = np.where(cmp["Sales_prev"]>0, cmp["WoW Sales"]/cmp["Sales_prev"], np.nan)
                    alerts = cmp[(cmp["Sales_prev"] >= min_prev_sales) & (cmp["%ΔSales"] <= pct)].copy()
                    return alerts.sort_values("%ΔSales").head(15)

                vendor_decl = _declines(cur_v.rename(columns={"Sales":"Sales_cur","Units":"Units_cur"}),
                                        prev_v.rename(columns={"Sales":"Sales_prev","Units":"Units_prev"}),
                                        key="Vendor", min_prev_sales=1000.0)
                retailer_decl = _declines(cur_r.rename(columns={"Sales":"Sales_cur","Units":"Units_cur"}),
                                          prev_r.rename(columns={"Sales":"Sales_prev","Units":"Units_prev"}),
                                          key="Retailer", min_prev_sales=5000.0)

                # Excel export
                excel_buf = io.BytesIO()
                with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
                    pd.DataFrame({
                        "Week": [_fmt_week_range(cur_start, cur_end)],
                        "Total Sales": [float(cur_s["Sales"].sum())],
                        "Total Units": [float(cur_s["Units"].sum())],
                        "New SKUs": [len(new_skus)],
                    }).to_excel(writer, sheet_name="Highlights", index=False)

                    movers.rename(columns={
                        "Sales_cur":"Sales (This Week)",
                        "Sales_prev":"Sales (Prev Week)",
                        "Units_cur":"Units (This Week)",
                        "Units_prev":"Units (Prev Week)",
                    }).to_excel(writer, sheet_name="Biggest Movers", index=False)

                    if not vendor_decl.empty:
                        vendor_decl.to_excel(writer, sheet_name="Declining Vendors", index=False)
                    if not retailer_decl.empty:
                        retailer_decl.to_excel(writer, sheet_name="Declining Retailers", index=False)
                    if new_skus:
                        pd.DataFrame({"New SKUs": new_skus}).to_excel(writer, sheet_name="New SKUs", index=False)
                    if mom is not None and not mom.empty:
                        mom.head(50).to_excel(writer, sheet_name="Momentum Top 50", index=False)

                excel_bytes = excel_buf.getvalue()
                st.download_button(
                    "Download Weekly Summary (Excel)",
                    data=excel_bytes,
                    file_name=f"Weekly_Summary_{pd.to_datetime(cur_end).date()}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key="dl_weekly_summary_excel",
                )

                # PDF export (simple highlights)
                # PDF export (professional)
                # KPIs
                total_sales_cur = float(cur_s["Sales"].sum())
                total_units_cur = float(cur_s["Units"].sum())
                total_sales_prev = float(prev_s["Sales"].sum()) if prev is not None and not prev.empty else 0.0
                total_units_prev = float(prev_s["Units"].sum()) if prev is not None and not prev.empty else 0.0

                kpis = [
                    ("Total Sales", f"${total_sales_cur:,.2f}", f"${total_sales_prev:,.2f}", f"${(total_sales_cur-total_sales_prev):,.2f}"),
                    ("Total Units", f"{int(round(total_units_cur)):,}", f"{int(round(total_units_prev)):,}", f"{int(round(total_units_cur-total_units_prev)):,}"),
                    ("New SKUs", f"{len(new_skus):,}", "-", "-"),
                ]

                # Top lists (with WoW Sales)
                # Current week base tables exist: cur_r/cur_v/cur_s; prior week: prev_r/prev_v/prev_s
                def _add_wow_sales(cur_df: pd.DataFrame, prev_df: pd.DataFrame, key: str) -> pd.DataFrame:
                    cur2 = cur_df.copy()
                    prev2 = prev_df.copy() if prev_df is not None else pd.DataFrame(columns=[key, "Sales", "Units"])
                    if key not in cur2.columns:
                        return cur2
                    if "Sales" not in cur2.columns:
                        cur2["Sales"] = 0.0
                    if "Units" not in cur2.columns:
                        cur2["Units"] = 0.0
                    if key not in prev2.columns:
                        prev2[key] = []
                    if "Sales" not in prev2.columns:
                        prev2["Sales"] = 0.0
                    prev2 = prev2[[key, "Sales"]].rename(columns={"Sales":"Sales_prev"})
                    cur2 = cur2.merge(prev2, on=key, how="left")
                    cur2["Sales_prev"] = cur2["Sales_prev"].fillna(0.0)
                    cur2["WoW $ Diff"] = cur2["Sales"] - cur2["Sales_prev"]
                    cur2["WoW $ %"] = np.where(cur2["Sales_prev"] > 0, cur2["WoW $ Diff"] / cur2["Sales_prev"], np.nan)
                    return cur2

                top_r = _add_wow_sales(cur_r, prev_r, "Retailer")
                top_v = _add_wow_sales(cur_v, prev_v, "Vendor")
                top_s = _add_wow_sales(cur_s, prev_s, "SKU")

                # Sort by current Sales and take Top N
                top_r = top_r.sort_values("Sales", ascending=False).head(10) if "Sales" in top_r.columns else top_r.head(10)
                top_v = top_v.sort_values("Sales", ascending=False).head(10) if "Sales" in top_v.columns else top_v.head(10)
                top_s = top_s.sort_values("Sales", ascending=False).head(10) if "Sales" in top_s.columns else top_s.head(10)

                # Format tables for PDF
                def _fmt_pdf_money(v):
                    try:
                        if isinstance(v, str) and v.strip().startswith("$"):
                            return v.strip()
                        return f"${float(v):,.2f}"
                    except:
                        return ""
                def _fmt_pdf_int(v):
                    try: return f"{int(round(float(v))):,}"
                    except: return ""

                # Column order for Page-1 Top sections: keep Units, Sales, add WoW Sales ($ and %)
                def _subset_top(df_in: pd.DataFrame, key_col: str) -> pd.DataFrame:
                    if df_in is None or df_in.empty:
                        return df_in
                    cols = [c for c in [key_col, "Units", "Sales", "WoW $ Diff"] if c in df_in.columns]
                    return df_in[cols]


                top_r_pdf = top_r.copy()
                if not top_r_pdf.empty:
                    # Keep Units column (current week) and format it
                    if "Units" in top_r_pdf.columns: top_r_pdf["Units"] = top_r_pdf["Units"].map(_fmt_pdf_int)
                    # Format Sales and WoW Sales
                    if "Sales" in top_r_pdf.columns: top_r_pdf["Sales"] = top_r_pdf["Sales"].map(_fmt_pdf_money)
                    if "WoW $ Diff" in top_r_pdf.columns: top_r_pdf["WoW $ Diff"] = top_r_pdf["WoW $ Diff"].map(_fmt_pdf_money)
                top_r_pdf = _subset_top(top_r_pdf, "Retailer")

                top_v_pdf = top_v.copy()
                if not top_v_pdf.empty:
                    # Keep Units column (current week) and format it
                    if "Units" in top_v_pdf.columns: top_v_pdf["Units"] = top_v_pdf["Units"].map(_fmt_pdf_int)
                    # Format Sales and WoW Sales
                    if "Sales" in top_v_pdf.columns: top_v_pdf["Sales"] = top_v_pdf["Sales"].map(_fmt_pdf_money)
                    if "WoW $ Diff" in top_v_pdf.columns: top_v_pdf["WoW $ Diff"] = top_v_pdf["WoW $ Diff"].map(_fmt_pdf_money)
                top_v_pdf = _subset_top(top_v_pdf, "Vendor")

                top_s_pdf = top_s.copy()
                if not top_s_pdf.empty:
                    # Keep Units column (current week) and format it
                    if "Units" in top_s_pdf.columns: top_s_pdf["Units"] = top_s_pdf["Units"].map(_fmt_pdf_int)
                    # Format Sales and WoW Sales
                    if "Sales" in top_s_pdf.columns: top_s_pdf["Sales"] = top_s_pdf["Sales"].map(_fmt_pdf_money)
                    if "WoW $ Diff" in top_s_pdf.columns: top_s_pdf["WoW $ Diff"] = top_s_pdf["WoW $ Diff"].map(_fmt_pdf_money)
                top_s_pdf = _subset_top(top_s_pdf, "SKU")

                movers_pdf = movers.rename(columns={
                    "Sales_cur":"Sales (This Week)",
                    "Sales_prev":"Sales (Prev Week)",
                    "Units_cur":"Units (This Week)",
                    "Units_prev":"Units (Prev Week)",
                }).copy()

                for c in ["Sales (This Week)","Sales (Prev Week)","WoW Sales"]:
                    if c in movers_pdf.columns: movers_pdf[c] = movers_pdf[c].map(_fmt_pdf_money)
                for c in ["Units (This Week)","Units (Prev Week)","WoW Units"]:
                    if c in movers_pdf.columns: movers_pdf[c] = movers_pdf[c].map(_fmt_pdf_int)

                vendor_decl_pdf = vendor_decl.copy() if "vendor_decl" in locals() else pd.DataFrame()
                retailer_decl_pdf = retailer_decl.copy() if "retailer_decl" in locals() else pd.DataFrame()

                # Format decline tables for PDF (currency/units/%)
                def _format_generic_pdf(df_in: pd.DataFrame) -> pd.DataFrame:
                    df_out = df_in.copy()
                    if df_out is None or df_out.empty:
                        return df_out
                    for col in df_out.columns:
                        lc = str(col).lower()
                        if "sales" in lc or lc.endswith("$"):
                            df_out[col] = df_out[col].map(lambda v: _fmt_pdf_money(v) if str(v) and not str(v).strip().startswith("$") else str(v))
                        elif "unit" in lc or "qty" in lc:
                            df_out[col] = df_out[col].map(lambda v: _fmt_pdf_int(v) if str(v) and str(v).replace(",","").replace(".","").isdigit() else str(v))
                        elif "%" in str(col) or "pct" in lc or "percent" in lc or "wow%" in lc:
                            def _pp(v):
                                try:
                                    vv=float(v)
                                    return f"{vv*100:,.1f}%"
                                except Exception:
                                    return str(v)
                            df_out[col] = df_out[col].map(_pp)
                    return df_out

                vendor_decl_pdf = _format_generic_pdf(vendor_decl_pdf)
                retailer_decl_pdf = _format_generic_pdf(retailer_decl_pdf)

                bullets = [
                    f"Week range: {_fmt_week_range(cur_start, cur_end)}",
                    f"Top Retailer (Sales): {top_r.iloc[0]['Retailer'] if not top_r.empty else '—'}",
                    f"Top Vendor (Sales): {top_v.iloc[0]['Vendor'] if not top_v.empty else '—'}",
                    f"Top SKU (Sales): {str(top_s.iloc[0]['SKU']) if not top_s.empty else '—'}",
                ]
                if new_skus:
                    bullets.append(f"New SKUs detected: {len(new_skus)}")

                subtitle = f"Weekly Summary • Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                sections = [
                    ("Highlights", None, bullets),
                    ("Top Retailers", top_r_pdf, None),
                    ("Top Vendors", top_v_pdf, None),
                    ("Top SKUs", top_s_pdf, None),
                    ("Biggest Movers (SKU)", movers_pdf.head(15), None),
                ]

                if vendor_decl_pdf is not None and not vendor_decl_pdf.empty:
                    sections.append(("Declining Vendors", vendor_decl_pdf.head(15).astype(str), None))
                if retailer_decl_pdf is not None and not retailer_decl_pdf.empty:
                    sections.append(("Declining Retailers", retailer_decl_pdf.head(15).astype(str), None))
                if new_skus:
                    sections.append(("New SKUs", pd.DataFrame({"New SKUs": new_skus[:50]}), None))
                if mom is not None and not mom.empty:
                    mom_pdf = mom.head(15).copy()
                    if "Momentum" in mom_pdf.columns:
                        mom_pdf["Momentum"] = mom_pdf["Momentum"].map(lambda x: f"{float(x):.0f}" if str(x)!="" else "")
                    if "Units_Last" in mom_pdf.columns:
                        mom_pdf["Units_Last"] = mom_pdf["Units_Last"].map(_fmt_pdf_int)
                    if "Sales_Last" in mom_pdf.columns:
                        mom_pdf["Sales_Last"] = mom_pdf["Sales_Last"].map(_fmt_pdf_money)
                    cols_mom = ["SKU","Momentum","Lookback Weeks","Up Weeks","Down Weeks","Weeks","Units_Last","Sales_Last"]
                    cols_mom = [c for c in cols_mom if c in mom_pdf.columns]
                    if cols_mom:
                        # Ensure numeric formatting for PDF
                        if "Units_Last" in mom_pdf.columns:
                            mom_pdf["Units_Last"] = mom_pdf["Units_Last"].map(_fmt_pdf_int)
                        if "Sales_Last" in mom_pdf.columns:
                            mom_pdf["Sales_Last"] = mom_pdf["Sales_Last"].map(_fmt_pdf_money)
                        # Momentum Leaders (PDF export uses last 12 weeks)
                mom_pdf = compute_momentum_table(df_all, window=12)
                if mom_pdf is not None and not mom_pdf.empty:
                    mom_pdf = mom_pdf.head(15).copy()

                    # Format numeric columns for PDF
                    if "Units_Last" in mom_pdf.columns:
                        mom_pdf["Units_Last"] = mom_pdf["Units_Last"].map(_fmt_pdf_int)
                    if "Sales_Last" in mom_pdf.columns:
                        mom_pdf["Sales_Last"] = mom_pdf["Sales_Last"].map(_fmt_pdf_money)

                    # Exact column order requested
                    mom_pdf = mom_pdf.rename(columns={"Momentum":"Momentum"})
                    cols_mom = ["SKU", "Momentum", "Up Weeks", "Down Weeks", "Units_Last", "Sales_Last"]
                    cols_mom = [c for c in cols_mom if c in mom_pdf.columns]
                    sections.append(("Momentum Leaders", mom_pdf[cols_mom].astype(str), None))
                # Build professional PDF (3-page executive layout)
                try:
                    mom_pdf_export = compute_momentum_table(d, window=12).head(15).copy()
                except Exception:
                    mom_pdf_export = pd.DataFrame(columns=["SKU","Momentum","Up Weeks","Down Weeks","Units_Last","Sales_Last"])

                if mom_pdf_export is None or mom_pdf_export.empty:
                    mom_pdf_export = pd.DataFrame(columns=["SKU","Momentum","Up Weeks","Down Weeks","Units_Last","Sales_Last"])
                else:
                    # ensure correct column names for PDF and formatting
                    if "Units_Last" in mom_pdf_export.columns: mom_pdf_export["Units_Last"] = mom_pdf_export["Units_Last"].map(_fmt_pdf_int)
                    if "Sales_Last" in mom_pdf_export.columns: mom_pdf_export["Sales_Last"] = mom_pdf_export["Sales_Last"].map(_fmt_pdf_money)


                # Highlights + KPI dict for PDF (safe defaults)
                total_sales_cur = float(cur["Sales"].sum()) if "Sales" in cur.columns else 0.0
                total_units_cur = float(cur["Units"].sum()) if "Units" in cur.columns else 0.0
                total_sales_prev = float(prev["Sales"].sum()) if (prev is not None and not prev.empty and "Sales" in prev.columns) else 0.0
                total_units_prev = float(prev["Units"].sum()) if (prev is not None and not prev.empty and "Units" in prev.columns) else 0.0

                wow_sales = total_sales_cur - total_sales_prev
                wow_units = total_units_cur - total_units_prev
                wow_sales_pct = (wow_sales / total_sales_prev) if total_sales_prev > 0 else None
                wow_units_pct = (wow_units / total_units_prev) if total_units_prev > 0 else None

                kpi_dict = {
                    "Sales": _fmt_pdf_money(total_sales_cur),
                    "Units": _fmt_pdf_int(total_units_cur),
                    "WoW Sales": _fmt_pdf_money(wow_sales) + (f" ({wow_sales_pct*100:,.1f}%)" if wow_sales_pct is not None else ""),
                    "WoW Units": _fmt_pdf_int(wow_units) + (f" ({wow_units_pct*100:,.1f}%)" if wow_units_pct is not None else ""),
                }

                # Simple highlight bullets (kept short for the PDF)
                highlights = [
                    f"Week {cur_start.date()} to {cur_end.date()} total sales {_fmt_pdf_money(total_sales_cur)} on {_fmt_pdf_int(total_units_cur)} units.",
                    f"WoW sales change: {_fmt_pdf_money(wow_sales)}.",
                    f"WoW units change: {_fmt_pdf_int(wow_units)}.",
                ]

                executive_takeaway, drivers_df, opportunities_df = _compute_wow_insights(d)

                pdf_bytes = make_weekly_summary_pdf_bytes(
                    "Weekly Summary",
                    highlights,
                    kpi_dict,
                    top_r_pdf,
                    top_v_pdf,
                    top_s_pdf,
                    movers_pdf if "movers_pdf" in locals() else pd.DataFrame(),
                    vendor_decl_pdf if "vendor_decl_pdf" in locals() else pd.DataFrame(),
                    retailer_decl_pdf if "retailer_decl_pdf" in locals() else pd.DataFrame(),
                    mom_pdf_export,
                    d,
                    logo_path=LOGO_PATH if "LOGO_PATH" in globals() else None,
                    executive_takeaway=executive_takeaway,
                    drivers_df=drivers_df,
                    opportunities_df=opportunities_df,
                )
                st.download_button(
                    "Download Weekly Summary (PDF)",
                    data=pdf_bytes,
                    file_name=f"Weekly_Summary_{pd.to_datetime(cur_end).date()}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="dl_weekly_summary_pdf",
                )

        st.divider()



def render_tab_comparisons():
    with tab_comparisons:
        st.subheader("Comparisons")

        view = st.selectbox("View", options=["Retailer / Vendor Comparison", "SKU Comparison"], index=0, key="cmp_view")

        if view == "Retailer / Vendor Comparison":
            render_comparison_retailer_vendor()
        else:
            render_comparison_sku()

        ctx = st.session_state.get("cmp_ctx", {})
        render_comparison_extras(ctx)



def render_tab_sku_intel():
    with tab_sku_intel:
        st.subheader("SKU Intelligence")
        view = st.selectbox("View", options=["SKU Health Score", "Lost Sales Detector"], index=0, key="sku_intel_view")
        if view == "SKU Health Score":
            render_sku_health()
        else:
            render_lost_sales()

        st.markdown("---")
        st.markdown("### SKU lookup")

        if df_all.empty:
            st.info("No sales data loaded.")
        else:
            dsl = df_all.copy()
            if "StartDate" in dsl.columns:
                dsl["StartDate"] = pd.to_datetime(dsl["StartDate"], errors="coerce")
                dsl["Year"] = dsl["StartDate"].dt.year
            else:
                dsl["Year"] = np.nan

            sku_opts = sorted([str(x).strip() for x in dsl.get("SKU", pd.Series([], dtype="object")).dropna().unique().tolist() if str(x).strip()])

            cL, cR = st.columns([2, 1])
            with cL:
                sku_query = st.text_input("Search SKU (type part of SKU)", value="", key="si_lookup_q")
            with cR:
                max_rows = st.selectbox("Max rows", options=[25, 50, 100, 200], index=1, key="si_lookup_max")

            if sku_query.strip():
                q = sku_query.strip().lower()
                matches = [s for s in sku_opts if q in s.lower()]
                if not matches:
                    st.warning("No SKUs match that search.")
                    matches = []
            else:
                matches = sku_opts[:200]

            pick_sku = st.selectbox("Select SKU", options=matches if matches else ["—"], index=0, key="si_lookup_pick")

            if pick_sku and pick_sku != "—":
                df_sku = dsl[dsl["SKU"].astype(str).str.strip() == str(pick_sku).strip()].copy()

                if df_sku.empty:
                    st.warning("No rows found for that SKU.")
                else:
                    units_total = float(df_sku["Units"].sum()) if "Units" in df_sku.columns else 0.0
                    sales_total = float(df_sku["Sales"].sum()) if "Sales" in df_sku.columns else 0.0
                    first_date = df_sku["StartDate"].min() if "StartDate" in df_sku.columns else None
                    last_date = df_sku["StartDate"].max() if "StartDate" in df_sku.columns else None

                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Total Units", fmt_int(units_total))
                    k2.metric("Total Sales", fmt_currency(sales_total))
                    k3.metric("First Week", first_date.date().isoformat() if pd.notna(first_date) else "—")
                    k4.metric("Last Week", last_date.date().isoformat() if pd.notna(last_date) else "—")

                    st.markdown("#### Breakdown by year")
                    by_year = df_sku.groupby("Year", as_index=False).agg(Units=("Units","sum"), Sales=("Sales","sum")).sort_values("Year")
                    by_year_disp = by_year.copy()
                    if "Units" in by_year_disp.columns:
                        by_year_disp["Units"] = by_year_disp["Units"].apply(fmt_int)
                    if "Sales" in by_year_disp.columns:
                        by_year_disp["Sales"] = by_year_disp["Sales"].apply(fmt_currency)
                    st.dataframe(by_year_disp, use_container_width=True, hide_index=True)

                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("#### Breakdown by retailer")
                        if "Retailer" in df_sku.columns:
                            by_r = df_sku.groupby("Retailer", as_index=False).agg(Units=("Units","sum"), Sales=("Sales","sum")).sort_values("Sales", ascending=False)
                            by_r_disp = by_r.copy()
                            by_r_disp["Units"] = by_r_disp["Units"].apply(fmt_int)
                            by_r_disp["Sales"] = by_r_disp["Sales"].apply(fmt_currency)
                            st.dataframe(style_numeric_posneg(by_r_disp.head(int(max_rows)), cols=[c for c in by_r_disp.columns if any(k in str(c).lower() for k in ['units','sales','delta','diff','change','%'])]), use_container_width=True, hide_index=True, height=_table_height(by_r_disp.head(int(max_rows)), max_px=450))
                        else:
                            st.write("—")
                    with c2:
                        st.markdown("#### Breakdown by vendor")
                        if "Vendor" in df_sku.columns:
                            by_v = df_sku.groupby("Vendor", as_index=False).agg(Units=("Units","sum"), Sales=("Sales","sum")).sort_values("Sales", ascending=False)
                            by_v_disp = by_v.copy()
                            by_v_disp["Units"] = by_v_disp["Units"].apply(fmt_int)
                            by_v_disp["Sales"] = by_v_disp["Sales"].apply(fmt_currency)
                            st.dataframe(style_numeric_posneg(by_v_disp.head(int(max_rows)), cols=[c for c in by_v_disp.columns if any(k in str(c).lower() for k in ['units','sales','delta','diff','change','%'])]), use_container_width=True, hide_index=True, height=_table_height(by_v_disp.head(int(max_rows)), max_px=450))
                        else:
                            st.write("—")

                    st.markdown("#### Weekly detail (most recent first)")
                    cols_show = [c for c in ["StartDate", "Retailer", "Vendor", "SKU", "Units", "Sales"] if c in df_sku.columns]
                    detail = df_sku[cols_show].copy()
                    if "StartDate" in detail.columns:
                        detail = detail.sort_values("StartDate", ascending=False)
                        detail["StartDate"] = detail["StartDate"].dt.date.astype(str)
                    if "Units" in detail.columns:
                        detail["Units"] = detail["Units"].apply(fmt_int)
                    if "Sales" in detail.columns:
                        detail["Sales"] = detail["Sales"].apply(fmt_currency)

                    detail = make_unique_columns(detail)
                    st.dataframe(style_numeric_posneg(detail.head(int(max_rows)), cols=[c for c in detail.columns if any(k in str(c).lower() for k in ['units','sales','delta','diff','change','%'])]), use_container_width=True, hide_index=True, height=_table_height(detail.head(int(max_rows)), max_px=650))



def render_tab_forecasting():
    with tab_forecasting:
        st.subheader("Forecasting")

        view = st.selectbox("View", options=["Run-Rate Forecast", "Seasonality"], index=0, key="fc_view")
        if view == "Run-Rate Forecast":
            render_runrate()
        else:
            render_seasonality()



def render_tab_alerts():
    with tab_alerts:
        st.subheader("Alerts")
        with st.expander("Data coverage (loaded history)", expanded=False):
            render_data_coverage_panel(df_all)
        view = st.selectbox("View", options=["Insights & Alerts", "No Sales SKUs"], index=0, key="alerts_view")
        if view == "Insights & Alerts":
            render_alerts()
        else:
            render_no_sales()



def render_tab_data_center():
    with tab_data_center:
        st.subheader("Data Center")

        # Coverage summary
        st.markdown("### Data coverage")
        render_data_coverage_panel(df_all)

        st.markdown("---")

        st.markdown("### Uploads")
        st.caption("All uploads live here (sidebar stays minimal).")

        # Bulk data upload (weekly + YOW workbooks)
        if "render_bulk_data_upload" in globals():
            render_bulk_data_upload()
        else:
            st.info("Bulk Data Upload tool not available in this build.")


def render_tab_admin():
    with tab_admin:
        st.subheader("Admin")

        if not edit_mode:
            st.info("Enable **Edit Mode** in the sidebar to access admin tools.")
            return

        tool = st.selectbox(
            "Admin tools",
            options=["Edit Vendor Map", "Backup / Restore", "Year Locks", "Cache Tools"],
            index=0,
            key="admin_tool",
        )

        if tool == "Edit Vendor Map":
            if "render_edit_vendor_map" in globals():
                render_edit_vendor_map()
            else:
                st.info("Vendor map editor not available in this build.")

        elif tool == "Backup / Restore":
            st.markdown("### Backup")
            st.caption("Download a full backup ZIP (data + config), or restore from a prior backup.")
            import io, zipfile, os

            def _safe_bytes(p):
                try:
                    return Path(p).read_bytes()
                except Exception:
                    return None

            # Build backup zip
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                # core files
                for fp in ["app.py", "requirements.txt", "cornerstone_logo.jpg"]:
                    if os.path.exists(fp):
                        z.write(fp, arcname=fp)

                # data folder
                if DATA_DIR.exists():
                    for root, _, files in os.walk(DATA_DIR):
                        for fn in files:
                            full = os.path.join(root, fn)
                            arc = os.path.relpath(full, start=os.getcwd())
                            z.write(full, arcname=arc)

            buf.seek(0)
            st.download_button(
                "Download full backup (ZIP)",
                data=buf.getvalue(),
                file_name="cornerstone_sales_app_backup.zip",
                mime="application/zip",
                use_container_width=True,
            )

            st.markdown("---")
            st.markdown("### Restore")
            st.caption("Upload a backup ZIP created here to restore the app's data/config.")

            up_zip = st.file_uploader("Upload backup ZIP", type=["zip"], key="admin_restore_zip")
            if st.button("Restore from ZIP", disabled=(up_zip is None), use_container_width=True, key="btn_restore_zip"):
                if up_zip is None:
                    st.warning("Upload a ZIP first.")
                else:
                    ok, msg = restore_app_backup_zip(up_zip.getvalue())
                    if ok:
                        st.cache_data.clear()
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

        elif tool == "Year Locks":
            st.markdown("### Year locks")
            st.caption("Lock closed years to prevent accidental overwrites during bulk uploads.")

            locked = load_year_locks()
            yrs = sorted(list(range(date.today().year - 5, date.today().year + 1)))
            sel = st.multiselect("Locked years", options=yrs, default=sorted(list(locked)))
            if st.button("Save year locks", use_container_width=True):
                save_year_locks(set(int(y) for y in sel))
                st.success("Saved.")
                st.rerun()

        elif tool == "Cache Tools":
            st.markdown("### Cache tools")
            st.caption("If something looks stale after a restore/upload, clear caches.")
            if st.button("Clear cached data", use_container_width=True):
                st.cache_data.clear()
                st.success("Cache cleared. Rerunning…")
                st.rerun()


def render_tab_year_summary():
    with tab_year_summary:
        st.subheader("Year Summary (YoY)")

        if df_all.empty:
            st.info("No sales data yet.")
        else:
            d = df_all.copy()
            d["StartDate"] = pd.to_datetime(d["StartDate"], errors="coerce")
            d = d[d["StartDate"].notna()].copy()
            d["Year"] = d["StartDate"].dt.year.astype(int)

            years = sorted(d["Year"].unique().tolist())
            current_year = int(max(years)) if years else None
            prior_year = int(sorted(years)[-2]) if len(years) >= 2 else None

            # Helper sums
            def _sum(df_, col):
                return float(df_[col].sum()) if (df_ is not None and not df_.empty and col in df_.columns) else 0.0

            def _pct(delta, base):
                return (delta / base) if base else np.nan

            # Default compare window for KPIs: last two years (current + prior)
            a = d[d["Year"] == int(prior_year)].copy() if prior_year is not None else d[d["Year"] == int(current_year)].copy()
            b = d[d["Year"] == int(current_year)].copy()

            # Basis toggle impacts driver + concentration calculations
            basis = st.radio("Basis (tables + drivers)", options=["Sales", "Units"], index=0, horizontal=True, key="ys_basis")
            value_col = "Sales" if basis == "Sales" else "Units"

            # =========================
            # KPIs (last 2 years)
            # =========================
            st.markdown("### KPIs (current year vs prior year, plus vs prior-years average)")

            uA, uB = _sum(a, "Units"), _sum(b, "Units")   # A=prior year, B=current year
            sA, sB = _sum(a, "Sales"), _sum(b, "Sales")
            uD, sD = uB - uA, sB - sA
            uP, sP = _pct(uD, uA), _pct(sD, sA)

            labelA = str(prior_year) if prior_year is not None else (str(current_year) if current_year is not None else "—")
            labelB = str(current_year) if current_year is not None else "—"

            # Current year vs all prior years (avg per year)
            prior_all = d[d["Year"] < int(current_year)].copy() if current_year is not None else d.iloc[0:0].copy()
            prior_years = sorted([y for y in years if y < int(current_year)]) if current_year is not None else []
            if prior_years:
                prior_by_year = prior_all.groupby("Year", as_index=False).agg(Units=("Units","sum"), Sales=("Sales","sum"))
                avg_units = float(prior_by_year["Units"].mean()) if not prior_by_year.empty else 0.0
                avg_sales = float(prior_by_year["Sales"].mean()) if not prior_by_year.empty else 0.0
            else:
                avg_units, avg_sales = 0.0, 0.0

            du_avg = uB - avg_units
            ds_avg = sB - avg_sales
            pu_avg = _pct(du_avg, avg_units)
            ps_avg = _pct(ds_avg, avg_sales)

            # 6 KPI cards = 6 datapoints
            k1, k2, k3, k4, k5, k6 = st.columns(6)

            # Units: current + prior
            k1.metric(f"Units ({labelB})", fmt_int(uB),
                      delta=(f"{fmt_int_signed(uD)} ({uP*100:.1f}%)" if (prior_year is not None and pd.notna(uP)) else (fmt_int_signed(uD) if prior_year is not None else None)))

            # Prior year card with inverse delta (so it reads as change to current)
            k2.metric(f"Units ({labelA})", fmt_int(uA))

            # Sales: current + prior
            k3.metric(f"Sales ({labelB})", fmt_currency(sB),
                      delta=(f"{fmt_currency_signed(sD)} ({sP*100:.1f}%)" if (prior_year is not None and pd.notna(sP)) else (fmt_currency_signed(sD) if prior_year is not None else None)))

            k4.metric(f"Sales ({labelA})", fmt_currency(sA))

            # Current vs all prior-years average
            k5.metric(f"Units vs prior-years avg ({len(prior_years)} yrs)", fmt_int(uB), delta=(f"{fmt_int_signed(du_avg)} ({pu_avg*100:.1f}%)" if (prior_years and pd.notna(pu_avg)) else (fmt_int_signed(du_avg) if prior_years else "—")))
            k6.metric(f"Sales vs prior-years avg ({len(prior_years)} yrs)", fmt_currency(sB), delta=(f"{fmt_currency_signed(ds_avg)} ({ps_avg*100:.1f}%)" if (prior_years and pd.notna(ps_avg)) else (fmt_currency_signed(ds_avg) if prior_years else "—")))


            # =========================
            # YoY driver breakdown (auto: prior year -> current year)
            # =========================
            st.markdown("### YoY driver breakdown (auto: prior year → current year)")

            if prior_year is None:
                st.info("Add at least two years of data to see YoY drivers.")
            else:
                sku_a = a.groupby("SKU", as_index=False).agg(Units_A=("Units","sum"), Sales_A=("Sales","sum"))
                sku_b = b.groupby("SKU", as_index=False).agg(Units_B=("Units","sum"), Sales_B=("Sales","sum"))
                sku = sku_a.merge(sku_b, on="SKU", how="outer").fillna(0.0)

                sku["A_val"] = sku["Sales_A"] if value_col == "Sales" else sku["Units_A"]
                sku["B_val"] = sku["Sales_B"] if value_col == "Sales" else sku["Units_B"]
                sku["Delta"] = sku["B_val"] - sku["A_val"]

                sku["Bucket"] = "Same (flat)"
                sku.loc[(sku["A_val"] == 0) & (sku["B_val"] > 0), "Bucket"] = "New SKUs"
                sku.loc[(sku["A_val"] > 0) & (sku["B_val"] == 0), "Bucket"] = "Lost SKUs"
                sku.loc[(sku["A_val"] > 0) & (sku["B_val"] > 0) & (sku["Delta"] > 0), "Bucket"] = "Same SKUs – Growth"
                sku.loc[(sku["A_val"] > 0) & (sku["B_val"] > 0) & (sku["Delta"] < 0), "Bucket"] = "Same SKUs – Decline"

                total_delta = float(sku["Delta"].sum())

                def _b(name):
                    return float(sku.loc[sku["Bucket"] == name, "Delta"].sum())

                b_new, b_lost = _b("New SKUs"), _b("Lost SKUs")
                b_grow, b_decl = _b("Same SKUs – Growth"), _b("Same SKUs – Decline")

                def _pct_of_delta(x):
                    return (x / total_delta) if total_delta else np.nan

                def _fmt(v):
                    return fmt_currency_signed(v) if value_col == "Sales" else fmt_int_signed(v)

                cD1, cD2, cD3, cD4 = st.columns(4)
                cD1.metric("New SKUs", _fmt(b_new), delta=(f"{_pct_of_delta(b_new)*100:.1f}%" if pd.notna(_pct_of_delta(b_new)) else "—"))
                cD2.metric("Lost SKUs", _fmt(b_lost), delta=(f"{_pct_of_delta(b_lost)*100:.1f}%" if pd.notna(_pct_of_delta(b_lost)) else "—"))
                cD3.metric("Same SKUs – Growth", _fmt(b_grow), delta=(f"{_pct_of_delta(b_grow)*100:.1f}%" if pd.notna(_pct_of_delta(b_grow)) else "—"))
                cD4.metric("Same SKUs – Decline", _fmt(b_decl), delta=(f"{_pct_of_delta(b_decl)*100:.1f}%" if pd.notna(_pct_of_delta(b_decl)) else "—"))

                with st.expander("Top SKU drivers", expanded=False):
                    tp = sku.sort_values("Delta", ascending=False).head(25).copy()
                    tn = sku.sort_values("Delta", ascending=True).head(25).copy()

                    tp2 = tp[["SKU","A_val","B_val","Delta","Bucket"]].rename(columns={"A_val": labelA, "B_val": labelB})
                    tn2 = tn[["SKU","A_val","B_val","Delta","Bucket"]].rename(columns={"A_val": labelA, "B_val": labelB})

                    st.markdown("**Top increases**")
                    tp2_disp = tp2.copy()
                    for c in [labelA, labelB, "Delta"]:
                        if c in tp2_disp.columns:
                            tp2_disp[c] = tp2_disp[c].apply(_fmt)
                    tp2_disp = make_unique_columns(tp2_disp)
                    st.dataframe(tp2_disp, use_container_width=True, height=_table_height(tp2_disp, max_px=700), hide_index=True)

                    st.markdown("**Top declines**")
                    tn2_disp = tn2.copy()
                    for c in [labelA, labelB, "Delta"]:
                        if c in tn2_disp.columns:
                            tn2_disp[c] = tn2_disp[c].apply(_fmt)
                    tn2_disp = make_unique_columns(tn2_disp)
                    st.dataframe(tn2_disp, use_container_width=True, height=_table_height(tn2_disp, max_px=700), hide_index=True)

            # =========================
            # Concentration risk (ALL YEARS, retailer + vendor)
            # =========================
            st.markdown("### Concentration risk (all years)")

            def _top_share(df_year, group_col, topn):
                g = df_year.groupby(group_col, as_index=False).agg(val=(value_col, "sum"))
                total = float(g["val"].sum())
                if total <= 0:
                    return 0.0
                return float(g.sort_values("val", ascending=False).head(topn)["val"].sum()) / total

            rows = []
            for y in years:
                dy = d[d["Year"] == int(y)].copy()
                rows.append({
                    "Year": int(y),
                    "Top 1 Retailer %": _top_share(dy, "Retailer", 1),
                    "Top 3 Retailers %": _top_share(dy, "Retailer", 3),
                    "Top 5 Retailers %": _top_share(dy, "Retailer", 5),
                    "Top 1 Vendor %": _top_share(dy, "Vendor", 1),
                    "Top 3 Vendors %": _top_share(dy, "Vendor", 3),
                    "Top 5 Vendors %": _top_share(dy, "Vendor", 5),
                })
            conc = pd.DataFrame(rows)
            conc_disp = conc.copy()
            try:
                st.dataframe(conc_disp.style.format({c: (lambda v: f"{v*100:.1f}%") for c in conc_disp.columns if c != "Year"}),
                             use_container_width=True, hide_index=True)
            except Exception:
                # fallback without Styler (rare streamlit/pyarrow edge cases)
                for c in [c for c in conc_disp.columns if c != "Year"]:
                    conc_disp[c] = conc_disp[c].apply(lambda v: f"{v*100:.1f}%")
                st.dataframe(conc_disp, use_container_width=True, hide_index=True)

            # =========================
            # Retailer summary (YEAR PICKER ONLY FOR THIS TABLE)
            # =========================
        
            st.markdown("#### Concentration breakdown (click to expand)")
            st.caption("Expand a year to see exactly which retailers/vendors make up the Top 1 / Top 3 / Top 5 shares.")

            def _top_list(df_year, group_col, topn):
                g = df_year.groupby(group_col, as_index=False).agg(val=(value_col, "sum")).sort_values("val", ascending=False)
                total = float(g["val"].sum())
                if total <= 0 or g.empty:
                    return g.assign(Share=0.0).head(0)
                g["Share"] = g["val"] / total
                return g.head(topn)

            for y in years[::-1]:  # newest first
                dy = d[d["Year"] == int(y)].copy()
                with st.expander(f"Year {int(y)} – show Top Retailers/Vendors", expanded=False):
                    cL, cR = st.columns(2)
                    with cL:
                        st.markdown("**Top Retailers**")
                        tr = _top_list(dy, "Retailer", 10)
                        if tr.empty:
                            st.write("—")
                        else:
                            tr_disp = tr.rename(columns={"val": value_col})
                            tr_disp[value_col] = tr_disp[value_col].apply(fmt_currency if value_col=="Sales" else fmt_int)
                            tr_disp["Share"] = tr_disp["Share"].apply(lambda v: f"{v*100:.1f}%")
                            st.dataframe(tr_disp[["Retailer", value_col, "Share"]], use_container_width=True, hide_index=True)
                            for n in [1, 3, 5]:
                                t = _top_list(dy, "Retailer", n)
                                if not t.empty:
                                    share = float(t["Share"].sum()) * 100
                                    names = ", ".join(t["Retailer"].astype(str).tolist())
                                    st.caption(f"Top {n}: {share:.1f}% — {names}")
                    with cR:
                        st.markdown("**Top Vendors**")
                        tv = _top_list(dy, "Vendor", 10)
                        if tv.empty:
                            st.write("—")
                        else:
                            tv_disp = tv.rename(columns={"val": value_col})
                            tv_disp[value_col] = tv_disp[value_col].apply(fmt_currency if value_col=="Sales" else fmt_int)
                            tv_disp["Share"] = tv_disp["Share"].apply(lambda v: f"{v*100:.1f}%")
                            st.dataframe(tv_disp[["Vendor", value_col, "Share"]], use_container_width=True, hide_index=True)
                            for n in [1, 3, 5]:
                                t = _top_list(dy, "Vendor", n)
                                if not t.empty:
                                    share = float(t["Share"].sum()) * 100
                                    names = ", ".join(t["Vendor"].astype(str).tolist())
                                    st.caption(f"Top {n}: {share:.1f}% — {names}")


            st.markdown("### Retailer summary")
            rs_year = st.selectbox("Retailer summary year", options=years, index=(len(years)-1), key="ys_rs_year")
            rs_prev = int(rs_year) - 1 if (int(rs_year) - 1) in years else None

            r0 = d[d["Year"] == int(rs_year)].groupby("Retailer", as_index=False).agg(Units=("Units","sum"), Sales=("Sales","sum"))
            if rs_prev is not None:
                r1 = d[d["Year"] == int(rs_prev)].groupby("Retailer", as_index=False).agg(Units_P=("Units","sum"), Sales_P=("Sales","sum"))
                r = r0.merge(r1, on="Retailer", how="left").fillna(0.0)
                r["Units Δ"] = r["Units"] - r["Units_P"]
                r["Sales Δ"] = r["Sales"] - r["Sales_P"]
            else:
                r = r0.copy()

            r = r.sort_values("Sales", ascending=False)
            r_show = r.copy()
            sty_r = r_show.style.format({
                'Units': fmt_int,
                'Sales': fmt_currency,
                'Units_P': fmt_int,
                'Sales_P': fmt_currency,
                'Units Δ': fmt_int_signed,
                'Sales Δ': fmt_currency_signed,
            })
            for c in ['Units Δ','Sales Δ']:
                if c in r_show.columns:
                    sty_r = sty_r.applymap(lambda v: _diff_color(v), subset=[c])
            r_show = make_unique_columns(r_show)
            st.dataframe(sty_r, use_container_width=True, height=_table_height(r_show, max_px=750), hide_index=True)

            # =========================
            # Vendor summary (YEAR PICKER ONLY FOR THIS TABLE)
            # =========================
            st.markdown("### Vendor summary")
            vs_year = st.selectbox("Vendor summary year", options=years, index=(len(years)-1), key="ys_vs_year")
            vs_prev = int(vs_year) - 1 if (int(vs_year) - 1) in years else None

            v0 = d[d["Year"] == int(vs_year)].groupby("Vendor", as_index=False).agg(Units=("Units","sum"), Sales=("Sales","sum"))
            if vs_prev is not None:
                v1 = d[d["Year"] == int(vs_prev)].groupby("Vendor", as_index=False).agg(Units_P=("Units","sum"), Sales_P=("Sales","sum"))
                vv = v0.merge(v1, on="Vendor", how="left").fillna(0.0)
                vv["Units Δ"] = vv["Units"] - vv["Units_P"]
                vv["Sales Δ"] = vv["Sales"] - vv["Sales_P"]
            else:
                vv = v0.copy()

            vv = vv.sort_values("Sales", ascending=False)
            vv_show = vv.copy()
            sty_v = vv_show.style.format({
                'Units': fmt_int,
                'Sales': fmt_currency,
                'Units_P': fmt_int,
                'Sales_P': fmt_currency,
                'Units Δ': fmt_int_signed,
                'Sales Δ': fmt_currency_signed,
            })
            for c in ['Units Δ','Sales Δ']:
                if c in vv_show.columns:
                    sty_v = sty_v.applymap(lambda v: _diff_color(v), subset=[c])
            vv_show = make_unique_columns(vv_show)
            st.dataframe(sty_v, use_container_width=True, height=_table_height(vv_show, max_px=750), hide_index=True)





def make_weekly_summary_pdf_bytes(title: str,
                                highlights: list,
                                kpi: dict,
                                top_retailers: pd.DataFrame,
                                top_vendors: pd.DataFrame,
                                top_skus: pd.DataFrame,
                                movers: pd.DataFrame,
                                vendor_decl: pd.DataFrame,
                                retailer_decl: pd.DataFrame,
                                momentum: pd.DataFrame,
                                df_all: pd.DataFrame,
                                logo_path: str = None,
                                executive_takeaway: str = None,
                                drivers_df: pd.DataFrame = None,
                                opportunities_df: pd.DataFrame = None) -> bytes:
    """
    Professional 3-page executive weekly PDF.
    Page 1: Executive Snapshot (KPIs + Performance Summary + Top tables)
    Page 2: Operational Movement (Trend chart + Biggest Movers + Declines)
    Page 3: Strategic Momentum (Momentum Leaders)
    """
    from io import BytesIO
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                    Image, KeepTogether, KeepInFrame, PageBreak)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.pdfgen import canvas as pdfcanvas
    from datetime import datetime
    import os as _os

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="H1", parent=styles["Heading1"], fontSize=16, leading=18, spaceAfter=8))
    styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], fontSize=12, leading=14, spaceAfter=6))
    styles.add(ParagraphStyle(name="Body", parent=styles["Normal"], fontSize=9.5, leading=12))
    styles.add(ParagraphStyle(name="Small", parent=styles["Normal"], fontSize=8, leading=10, textColor=colors.HexColor("#6b7280")))

    def _wow_color(v):
        """Green for positive, red for negative, neutral for zero/unknown."""
        try:
            s = str(v)
            import re as _re
            s2 = _re.sub(r"[^0-9\-\.]", "", s)
            if s2 in ("", "-", ".", "-."):
                raise ValueError("no number")
            x = float(s2)
            if x > 0:
                return colors.HexColor("#2ecc71")
            if x < 0:
                return colors.HexColor("#e74c3c")
        except Exception:
            pass
        return colors.HexColor("#111827")

    def _make_table(df: pd.DataFrame, header_bg="#111827", max_rows=10, col_widths=None,
                    wow_col_name=None, right_align_cols=None):
        if df is None or df.empty:
            return Paragraph("No data.", styles["Body"])
        tshow = df.copy().head(max_rows)

        # Format numeric columns intelligently (Sales = currency, Units = integers)
        disp = tshow.copy()
        for c in disp.columns:
            col_name = str(c).lower()
            ser = disp[c]
            num = pd.to_numeric(ser, errors="coerce")
            is_num = num.notna().any()
            if not is_num:
                disp[c] = ser.astype(str)
                continue

            def _fmt_currency(v):
                try:
                    v = float(v)
                except Exception:
                    return "—"
                return f"-${abs(v):,.2f}" if v < 0 else f"${v:,.2f}"

            def _fmt_int(v):
                try:
                    return f"{int(round(float(v))):,}"
                except Exception:
                    return "—"

            if ("sale" in col_name) or ("revenue" in col_name) or ("$" in col_name):
                disp[c] = num.map(_fmt_currency)
            elif ("unit" in col_name) or ("qty" in col_name) or ("quantity" in col_name):
                disp[c] = num.map(_fmt_int)
            else:
                def _fmt_generic(v):
                    # Robust numeric formatting: handles strings like "$1,234.56", "(123.45)", "—"
                    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                        return "—"
                    # pandas NA
                    try:
                        if pd.isna(v):
                            return "—"
                    except Exception:
                        pass
                    # If already numeric, keep
                    vv = v
                    if isinstance(vv, str):
                        s = vv.strip()
                        if s in ("", "—", "-", "N/A", "NA", "nan"):
                            return "—"
                        neg = False
                        if s.startswith("(") and s.endswith(")"):
                            neg = True
                            s = s[1:-1]
                        s = s.replace("$", "").replace(",", "").strip()
                        try:
                            vv = float(s)
                            if neg:
                                vv = -vv
                        except Exception:
                            return str(v)
                    try:
                        vv = float(vv)
                    except Exception:
                        return str(v)
                    if not math.isfinite(vv):
                        return "—"
                    if abs(vv - round(vv)) < 1e-9:
                        return f"{int(round(vv)):,}"
                    return f"{vv:,.2f}"


        data = [disp.columns.tolist()] + disp.values.tolist()
        tbl = Table(data, hAlign="LEFT", colWidths=col_widths)

        base = [
            ("BACKGROUND",(0,0),(-1,0), colors.HexColor(header_bg)),
            ("TEXTCOLOR",(0,0),(-1,0), colors.white),
            ("FONTNAME",(0,0),(-1,0), "Helvetica-Bold"),
            ("FONTSIZE",(0,0),(-1,0), 9),
            ("GRID",(0,0),(-1,-1), 0.25, colors.HexColor("#d1d5db")),
            ("FONTNAME",(0,1),(-1,-1), "Helvetica"),
            ("FONTSIZE",(0,1),(-1,-1), 8),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, colors.HexColor("#f9fafb")]),
            ("VALIGN",(0,0),(-1,-1), "TOP"),
            ("LEFTPADDING",(0,0),(-1,-1), 4),
            ("RIGHTPADDING",(0,0),(-1,-1), 4),
            ("TOPPADDING",(0,0),(-1,-1), 3),
            ("BOTTOMPADDING",(0,0),(-1,-1), 3),
        ]

        if right_align_cols:
            for col in right_align_cols:
                if col in tshow.columns:
                    j = tshow.columns.get_loc(col)
                    base.append(("ALIGN",(j,1),(j,-1),"RIGHT"))
                    base.append(("ALIGN",(j,0),(j,0),"RIGHT"))

        if wow_col_name and wow_col_name in tshow.columns:
            j = tshow.columns.get_loc(wow_col_name)
            for i in range(1, len(data)):
                base.append(("TEXTCOLOR",(j,i),(j,i), _wow_color(data[i][j])))
                base.append(("FONTNAME",(j,i),(j,i), "Helvetica-Bold"))

        tbl.setStyle(TableStyle(base))
        return tbl

    def _make_trend_chart(df_all: pd.DataFrame):
        if df_all is None or df_all.empty or "EndDate" not in df_all.columns:
            return None
        d = df_all.copy()
        d["EndDate"] = pd.to_datetime(d["EndDate"], errors="coerce")
        d = d[d["EndDate"].notna()].copy()
        if d.empty:
            return None
        wk = d.groupby("EndDate", as_index=False)[["Sales"]].sum().sort_values("EndDate").tail(12)
        if wk.empty:
            return None
        fig = plt.figure(figsize=(6.4, 2.2))
        ax = fig.add_subplot(111)
        ax.plot(wk["EndDate"], wk["Sales"])
        ax.set_title("Total Sales – Last 12 Weeks")
        ax.set_ylabel("Sales ($)")
        ax.tick_params(axis='x', labelrotation=45)
        ax.grid(True, linewidth=0.5, alpha=0.4)
        fig.tight_layout()
        img_buf = BytesIO()
        fig.savefig(img_buf, format="png", dpi=150)
        plt.close(fig)
        img_buf.seek(0)
        return img_buf

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    def _on_page(canv: pdfcanvas.Canvas, doc):
        canv.saveState()
        w, h = letter
        canv.setFillColor(colors.HexColor("#111827"))
        canv.rect(0, h-0.65*inch, w, 0.65*inch, fill=1, stroke=0)

        if logo_path and _os.path.exists(logo_path):
            try:
                canv.drawImage(logo_path, 0.55*inch, h-0.60*inch, width=1.3*inch, height=0.45*inch, mask='auto', preserveAspectRatio=True, anchor='sw')
            except Exception:
                pass

        canv.setFillColor(colors.white)
        canv.setFont("Helvetica-Bold", 12)
        canv.drawString(2.1*inch, h-0.40*inch, title)

        canv.setFont("Helvetica", 9)
        canv.drawRightString(w-0.55*inch, h-0.40*inch, f"Generated: {generated}")

        canv.setFillColor(colors.HexColor("#6b7280"))
        canv.setFont("Helvetica", 8)
        canv.drawString(0.55*inch, 0.45*inch, "Cornerstone Products Group – Confidential (Internal Use Only)")
        canv.drawRightString(w-0.55*inch, 0.45*inch, f"Page {doc.page}")
        canv.restoreState()

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
                            leftMargin=0.55*inch, rightMargin=0.55*inch,
                            topMargin=0.85*inch, bottomMargin=0.7*inch)
    story = []

    # PAGE 1
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("Executive Snapshot", styles["H1"]))

    # KPI panel
    kpi_lines = []
    if isinstance(kpi, dict) and kpi:
        for k, v in kpi.items():
            kpi_lines.append([str(k), str(v)])
    if not kpi_lines:
        kpi_lines = [["Sales", "—"], ["Units", "—"], ["WoW Sales", "—"], ["WoW Units", "—"]]
    kpi_tbl = Table(kpi_lines, colWidths=[2.2*inch, 1.6*inch])
    kpi_tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1), colors.HexColor("#f3f4f6")),
        ("BOX",(0,0),(-1,-1), 0.5, colors.HexColor("#d1d5db")),
        ("INNERGRID",(0,0),(-1,-1), 0.25, colors.HexColor("#e5e7eb")),
        ("FONTNAME",(0,0),(0,-1), "Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1), 9),
        ("LEFTPADDING",(0,0),(-1,-1), 8),
        ("RIGHTPADDING",(0,0),(-1,-1), 8),
        ("TOPPADDING",(0,0),(-1,-1), 6),
        ("BOTTOMPADDING",(0,0),(-1,-1), 6),
    ]))

    perf_bullets = []
    if highlights:
        for b in highlights[:6]:
            perf_bullets.append(f"• {b}")
    if not perf_bullets:
        perf_bullets = ["• No highlights generated for this week."]
    perf_par = Paragraph("<br/>".join(perf_bullets), styles["Body"])
    perf_box = Table([[Paragraph("Performance Summary", styles["H2"])],[perf_par]], colWidths=[3.9*inch])
    perf_box.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,1), colors.HexColor("#f9fafb")),
        ("BOX",(0,0),(-1,1), 0.5, colors.HexColor("#d1d5db")),
        ("LEFTPADDING",(0,0),(-1,-1), 10),
        ("RIGHTPADDING",(0,0),(-1,-1), 10),
        ("TOPPADDING",(0,0),(-1,-1), 8),
        ("BOTTOMPADDING",(0,0),(-1,-1), 8),
    ]))
    top_row = Table([[kpi_tbl, perf_box]], colWidths=[2.0*inch, doc.width-2.0*inch])
    top_row.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1), "TOP"), ("LEFTPADDING",(0,0),(-1,-1), 0), ("RIGHTPADDING",(0,0),(-1,-1), 0)]))
    story.append(top_row)
    story.append(Spacer(1, 0.14*inch))
    if executive_takeaway:
        story.append(Spacer(1, 0.06*inch))
        takeaway_box = Table([[Paragraph("<b>Executive takeaway:</b> " + html.escape(str(executive_takeaway)), styles["Body"])]],
                             colWidths=[doc.width])
        takeaway_box.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,-1), colors.HexColor("#f9fafb")),
            ("BOX",(0,0),(-1,-1), 0.5, colors.HexColor("#d1d5db")),
            ("LEFTPADDING",(0,0),(-1,-1), 10),
            ("RIGHTPADDING",(0,0),(-1,-1), 10),
            ("TOPPADDING",(0,0),(-1,-1), 8),
            ("BOTTOMPADDING",(0,0),(-1,-1), 8),
        ]))
        story.append(takeaway_box)
        story.append(Spacer(1, 0.10*inch))


    left_stack = [Paragraph("Top Retailers", styles["H2"]),
                  _make_table(top_retailers, max_rows=5, wow_col_name="WoW $ Diff", right_align_cols=["Units","Sales","WoW $ Diff"]),
                  Spacer(1, 0.12*inch),
                  Paragraph("Top Vendors", styles["H2"]),
                  _make_table(top_vendors, max_rows=5, wow_col_name="WoW $ Diff", right_align_cols=["Units","Sales","WoW $ Diff"])]
    right_stack = [Paragraph("Top SKUs", styles["H2"]),
                   _make_table(top_skus, max_rows=10, wow_col_name="WoW $ Diff", right_align_cols=["Units","Sales","WoW $ Diff"])]

    gutter = 0.18*inch
    left_w = (doc.width - gutter) * 0.49
    right_w = (doc.width - gutter) * 0.51
    left_cell = KeepInFrame(left_w, 6.6*inch, left_stack, mode="shrink")
    right_cell = KeepInFrame(right_w, 6.6*inch, right_stack, mode="shrink")
    split = Table([[left_cell, "", right_cell]], colWidths=[left_w, gutter, right_w])
    split.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"), ("LEFTPADDING",(0,0),(-1,-1), 0), ("RIGHTPADDING",(0,0),(-1,-1), 0)]))
    story.append(split)

    story.append(PageBreak())

    # PAGE 2
    story.append(Paragraph("Operational Movement", styles["H1"]))

    # Top 3 drivers of WoW sales change + Top opportunities
    if drivers_df is not None and hasattr(drivers_df, "empty") and (not drivers_df.empty):
        story.append(Spacer(1, 0.06*inch))
        story.append(KeepTogether([Paragraph("Top 3 Drivers of WoW Sales Change", styles["H2"]),
                                   _make_table(drivers_df, max_rows=3, wow_col_name="WoW Sales ($)",
                                               right_align_cols=["WoW Sales ($)","Sales (This Week)","Units (This Week)","WoW Units"])]))
        story.append(Spacer(1, 0.12*inch))

    if opportunities_df is not None and hasattr(opportunities_df, "empty") and (not opportunities_df.empty):
        story.append(KeepTogether([Paragraph("Top Opportunities (Positive WoW)", styles["H2"]),
                                   _make_table(opportunities_df, max_rows=8, wow_col_name="WoW Sales ($)",
                                               right_align_cols=["WoW Sales ($)","Sales (This Week)","Units (This Week)","WoW Units"])]))
        story.append(Spacer(1, 0.14*inch))

    story.append(Spacer(1, 0.08*inch))

    chart_buf = _make_trend_chart(df_all)
    if chart_buf is not None:
        img = Image(chart_buf, width=doc.width, height=2.0*inch)
        chart_box = Table([[img]], colWidths=[doc.width])
        chart_box.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,-1), colors.HexColor("#f9fafb")),
            ("BOX",(0,0),(-1,-1), 0.5, colors.HexColor("#d1d5db")),
            ("LEFTPADDING",(0,0),(-1,-1), 6),
            ("RIGHTPADDING",(0,0),(-1,-1), 6),
            ("TOPPADDING",(0,0),(-1,-1), 6),
            ("BOTTOMPADDING",(0,0),(-1,-1), 6),
        ]))
        story.append(chart_box)
        story.append(Spacer(1, 0.14*inch))

    story.append(KeepTogether([Paragraph("Biggest Movers", styles["H2"]), _make_table(movers, max_rows=12)]))
    story.append(Spacer(1, 0.14*inch))

    story.append(KeepTogether([Paragraph("Declining Vendors", styles["H2"]), _make_table(vendor_decl, max_rows=10, wow_col_name="WoW Sales")]))
    story.append(Spacer(1, 0.12*inch))
    story.append(KeepTogether([Paragraph("Declining Retailers", styles["H2"]), _make_table(retailer_decl, max_rows=10, wow_col_name="WoW Sales")]))

    story.append(PageBreak())

    # PAGE 3
    story.append(Paragraph("Strategic Momentum", styles["H1"]))
    story.append(Paragraph("Momentum Leaders (Last 12 Weeks)", styles["H2"]))
    story.append(Spacer(1, 0.06*inch))
    mom_flow = [_make_table(momentum, max_rows=15, right_align_cols=["Momentum","Up Weeks","Down Weeks","Units_Last","Sales_Last"])]
    story.append(KeepInFrame(doc.width, 8.5*inch, mom_flow, mode="shrink"))

    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    return buf.getvalue()


def make_simple_pdf_bytes(title: str, lines: list[str], table_df: pd.DataFrame|None = None) -> bytes:
    """Simple PDF generator for weekly summaries."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import inch
    except Exception:
        body = title + "\n\n" + "\n".join(lines or [])
        return body.encode("utf-8", errors="ignore")

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    w, h = letter
    x = 0.75 * inch
    y = h - 0.9 * inch

    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, title)
    y -= 0.4 * inch
    c.setFont("Helvetica", 10)

    for ln in (lines or []):
        if y < 0.75 * inch:
            c.showPage()
            y = h - 0.9 * inch
            c.setFont("Helvetica", 10)
        c.drawString(x, y, str(ln)[:180])
        y -= 0.22 * inch

    c.showPage()
    c.save()
    return buf.getvalue()


render_tab_overview()

render_tab_action_center()

render_tab_momentum()


render_tab_totals_dash()

render_tab_top_skus()

render_tab_wow_exc()

render_tab_exec()

render_tab_comparisons()

render_tab_sku_intel()

render_tab_forecasting()

render_tab_alerts()

render_tab_data_center()

render_tab_admin()

render_tab_year_summary()