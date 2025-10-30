# app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime
import os, io, re, requests

# ================== PAGE CONFIG ==================
st.set_page_config(layout="wide", page_title="Business & Medical Analytics Dashboard")
st.title("📊 Business & Medical Analytics Dashboard")

# ================== DATA SOURCE ==================
DATA_PATH = os.getenv("DATA_URL", "final_test_data_20250529.parquet")

def _is_http_url(s: str) -> bool:
    return bool(re.match(r"^https?://", str(s or ""), re.IGNORECASE))

@st.cache_data(ttl=3600, show_spinner="กำลังโหลดข้อมูล...")
def load_parquet(source: str) -> pd.DataFrame:
    if _is_http_url(source):
        r = requests.get(source, timeout=180)
        r.raise_for_status()
        return pd.read_parquet(io.BytesIO(r.content))
    return pd.read_parquet(source)

try:
    df_raw = load_parquet(DATA_PATH)
except Exception as e:
    st.error(f"โหลดข้อมูลไม่สำเร็จ: {e}")
    st.stop()

# ================== CLEANING ==================
def clean_dataframe(df: pd.DataFrame, strict_dedup: bool = True) -> pd.DataFrame:
    df = df.copy()

    # --- unify description
    desc_alias = [c for c in ["Description","Dscription","dscription","description"] if c in df.columns]
    if desc_alias:
        if desc_alias[0] != "Description":
            df.rename(columns={desc_alias[0]: "Description"}, inplace=True)

    # --- common types
    if "Posting Date" in df.columns:
        df["Posting Date"] = pd.to_datetime(df["Posting Date"], errors="coerce")

    # try to parse Payment/Due/Invoice related dates if exist
    for dcol in ["Payment Date", "Paid Date", "Due Date", "Invoice Date", "Document Date"]:
        if dcol in df.columns:
            df[dcol] = pd.to_datetime(df[dcol], errors="coerce")

    for c in ["LineTotal","avg_cost","age","Quantity"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["Branch","เพศ คนไข้","โรงพยาบาล","Customer/Vendor Name","group_disease","Description"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # fill numeric nans
    for c in ["LineTotal","avg_cost","age","Quantity"]:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # drop rows with missing date
    if "Posting Date" in df.columns:
        df = df[~df["Posting Date"].isna()].copy()
        df["Year"] = df["Posting Date"].dt.year
        df["YM"]   = df["Posting Date"].dt.to_period("M").astype(str)

    # gender / age group / disease
    gender_mapping = {"M":"Male","F":"Female","W":"Female","ชาย":"Male","หญิง":"Female"}
    if "เพศ คนไข้" in df.columns:
        df["gender_mapped"] = df["เพศ คนไข้"].map(gender_mapping).fillna("Other")

    if "age" in df.columns:
        bins   = [0,17,35,55,200]
        labels = ["0–17","18–35","36–55","56+"]
        df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=True)

    disease_mapping = {
        "กล้ามเนื้อเคล็ด":"Muscle Strain",
        "โรคทางเดินปัสสาวะ":"Urinary Tract Disease",
        "ปัจจัยที่มีผลต่อสถานะสุขภาพ":"Factors Affecting Health Status",
        "ความผิดปกติจากทางคลินิกและห้องปฏิบัติการ":"Abnormalities from Clinical",
        "โรคทางเดินอาหาร":"Gastrointestinal Disease",
        "URI":"Upper Respiratory Infection (URI)",
        "การติดเชื้อไวรัส":"Viral Infection",
        "การบาดเจ็บ การเป็นพิษ และอุบัติเหตุ":"Injury, Poisoning, and Accidents",
    }
    if "group_disease" in df.columns:
        df["disease_group_mapped"] = df["group_disease"].map(disease_mapping).fillna(df["group_disease"])

    # --- normalize "สิทธิการรักษา / ผู้ชำระเงิน / Payer"
    payer_aliases = [
        "Customer/Vendor Name", "สิทธิการรักษา", "ผู้ชำระเงิน", "Payer", "Insurance", "สิทธิ์การรักษา"
    ]
    payer_col = next((c for c in payer_aliases if c in df.columns), None)
    if payer_col and payer_col != "Customer/Vendor Name":
        df.rename(columns={payer_col: "Customer/Vendor Name"}, inplace=True)

    # --- normalize "วิธีจ่ายเงิน / Payment Method"
    paym_aliases = [
        "Payment Method","วิธีชำระเงิน","ช่องทางชำระเงิน","ประเภทการชำระเงิน","Payment Type","Method"
    ]
    paym_col = next((c for c in paym_aliases if c in df.columns), None)
    if paym_col:
        if paym_col != "Payment Method":
            df.rename(columns={paym_col: "Payment Method"}, inplace=True)
        df["Payment Method"] = df["Payment Method"].astype(str).str.strip()

    # --- strict de-dup
    candidate_keys = [
        ["Branch","Posting Date","Document No","Line No","LineTotal"],
        ["Branch","Posting Date","Document No","Description","LineTotal"],
        ["Branch","Posting Date","Description","LineTotal"],
    ]
    if strict_dedup:
        used_key = None
        before = len(df)
        for ks in candidate_keys:
            ks = [k for k in ks if k in df.columns]
            if len(ks) >= 3:
                dup_mask = df.duplicated(subset=ks, keep=False)
                dup_cnt  = int(dup_mask.sum())
                if dup_cnt > 0:
                    df = df.drop_duplicates(subset=ks)
                    used_key = ks
                    break
        after = len(df)
        if used_key is not None and after < before:
            st.warning(f"🧹 ลบข้อมูลซ้ำ {before-after:,} แถว ด้วยคีย์ {used_key}")
        elif used_key is None:
            dup_any = df.duplicated(keep=False).sum()
            if dup_any > 0:
                st.warning(f"พบรูปแบบซ้ำ {dup_any:,} แถว แต่ไม่มีคีย์ที่เหมาะสม — กรุณาตรวจคีย์เอกลักษณ์")

    return df

# เปิด/ปิด strict de-dup ใน sidebar
st.sidebar.header("⚙️ Filters")
strict_dedup = st.sidebar.checkbox("Strict de-dup (ตัดแถวซ้ำอัตโนมัติ)", value=True)

df = clean_dataframe(df_raw, strict_dedup=strict_dedup)

# ================== PROFIT & FILTERS ==================
profit_formula = st.sidebar.selectbox(
    "สูตรคำนวณกำไร",
    (
        "Per-Unit Cost: LineTotal - (avg_cost × Quantity)",
        "Current: LineTotal - avg_cost",
        "Fixed 40% Margin: LineTotal × 0.40",
    ),
    help="เลือกสูตรกำไรสำหรับทุกกราฟ/ตาราง",
)

df = df.copy()
if profit_formula.startswith("Per-Unit"):
    if {"avg_cost","Quantity"}.issubset(df.columns):
        df["Profit"] = df["LineTotal"] - (df["avg_cost"] * df["Quantity"])
    else:
        df["Profit"] = 0.0
elif profit_formula.startswith("Current"):
    df["Profit"] = df["LineTotal"] - df.get("avg_cost", 0)
else:
    df["Profit"] = df["LineTotal"] * 0.40

min_date = df["Posting Date"].min().date()
max_date = df["Posting Date"].max().date()
date_range = st.sidebar.date_input("ช่วงวันที่", (min_date, max_date), min_value=min_date, max_value=max_date)

branch_list = sorted(df["Branch"].dropna().unique().tolist())
branch_mode = st.sidebar.radio("โหมดเลือกสาขา", ["ทุกสาขา","เลือกบางสาขา"], horizontal=True)
if branch_mode == "เลือกบางสาขา":
    selected_branches = st.sidebar.multiselect("เลือกสาขา", branch_list, default=branch_list[:10])
else:
    selected_branches = branch_list

start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
df_filtered = df[
    (df["Posting Date"] >= start_date) &
    (df["Posting Date"] <= end_date) &
    (df["Branch"].isin(selected_branches))
].copy()

if df_filtered.empty:
    st.warning("ไม่พบข้อมูลตามตัวกรอง")
    st.stop()

st.caption(f"🧮 Using profit formula: {profit_formula}")
st.info(f"สาขาที่กำลังแสดงผล: {', '.join(selected_branches[:15])}{' ...' if len(selected_branches)>15 else ''}  • ช่วงวันที่: {start_date.date()} → {end_date.date()}")

# ================== TABS ==================
biz_tab, med_tab = st.tabs(["📈 Business","🩺 Medical"])

# ================== BUSINESS ANALYTICS ==================
with biz_tab:
    st.markdown("## 📈 Business Analytics")

    # SUMMARY OVERVIEW
    total_revenue = df_filtered["LineTotal"].sum()
    total_profit  = df_filtered["Profit"].sum()
    avg_margin = (total_profit / total_revenue * 100) if total_revenue else 0
    branch_count = df_filtered["Branch"].nunique()
    year_range = f"{df_filtered['Posting Date'].dt.year.min()}–{df_filtered['Posting Date'].dt.year.max()}"

    st.markdown(f"""
    <div style="display:flex; gap:1.5rem; flex-wrap:wrap; margin-bottom:1.5rem;">
      <div style="flex:1; min-width:220px; background:#1c1c1c; padding:1rem; border-radius:0.8rem; text-align:center;">
        <h4 style="margin:0;color:#ccc;">💰 รายได้รวมทั้งหมด</h4>
        <h2 style="margin:0;color:#52c41a;">{total_revenue:,.0f} ฿</h2>
      </div>
      <div style="flex:1; min-width:220px; background:#1c1c1c; padding:1rem; border-radius:0.8rem; text-align:center;">
        <h4 style="margin:0;color:#ccc;">📊 กำไรขั้นต้นรวม</h4>
        <h2 style="margin:0;color:#fadb14;">{total_profit:,.0f} ฿</h2>
      </div>
      <div style="flex:1; min-width:220px; background:#1c1c1c; padding:1rem; border-radius:0.8rem; text-align:center;">
        <h4 style="margin:0;color:#ccc;">📈 อัตรากำไรเฉลี่ย</h4>
        <h2 style="margin:0;color:#1890ff;">{avg_margin:.1f}%</h2>
      </div>
      <div style="flex:1; min-width:220px; background:#1c1c1c; padding:1rem; border-radius:0.8rem; text-align:center;">
        <h4 style="margin:0;color:#ccc;">🏢 จำนวนสาขา</h4>
        <h2 style="margin:0;color:#fff;">{branch_count:,}</h2>
      </div>
      <div style="flex:1; min-width:220px; background:#1c1c1c; padding:1rem; border-radius:0.8rem; text-align:center;">
        <h4 style="margin:0;color:#ccc;">📅 ปีข้อมูล</h4>
        <h2 style="margin:0;color:#fff;">{year_range}</h2>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Top 5 Branches by Revenue and Profit
    st.subheader("Top 5 Branches by Revenue and Profit")
    top5 = (
        df_filtered.groupby("Branch")[ ["LineTotal","Profit"] ].sum()
          .sort_values("LineTotal", ascending=False)
          .head(5)
          .reset_index()
    )
    if top5.empty:
        st.info("ไม่มีข้อมูลสำหรับกราฟนี้")
    else:
        top5_melt = top5.melt(id_vars="Branch", value_vars=["LineTotal","Profit"],
                              var_name="Metric", value_name="Value")
        top5_melt["Metric"] = top5_melt["Metric"].astype("category")
        top5_melt["Value"]  = pd.to_numeric(top5_melt["Value"], errors="coerce").fillna(0)

        chart = (
            alt.Chart(top5_melt)
            .mark_bar()
            .encode(
                x=alt.X("Branch:N", title="Branch"),
                y=alt.Y("Value:Q", title="Amount (฿)"),
                color=alt.Color("Metric:N", title="Metric", scale=alt.Scale(scheme="tableau10")),
                tooltip=["Branch","Metric",alt.Tooltip("Value:Q",format=",.0f",title="Amount (฿)")]
            )
        )
        st.altair_chart(chart, use_container_width=True)

    # Product Revenue Contribution (Top 10)
    st.subheader("Product Revenue Contribution (Top 10)")
    if "Description" in df_filtered.columns:
        prod = (df_filtered.groupby("Description")["LineTotal"].sum()
                .reset_index()
                .sort_values("LineTotal", ascending=False).head(10))
        total_sum = prod["LineTotal"].sum()
        prod["Percent"] = (prod["LineTotal"]/total_sum)*100 if total_sum>0 else 0

        base = alt.Chart(prod).encode(
            theta=alt.Theta("LineTotal:Q", stack=True),
            color=alt.Color("Description:N", title="Product/Service"),
            tooltip=[
                "Description:N",
                alt.Tooltip("LineTotal:Q", title="Revenue", format=",.0f"),
                alt.Tooltip("Percent:Q", title="Percent (%)", format=".1f"),
            ],
        )
        st.altair_chart(
            base.mark_arc(outerRadius=120) +
            base.mark_text(radius=145, size=12).encode(text=alt.Text("Percent:Q", format=".1f")),
            use_container_width=True
        )
    else:
        st.info("ไม่พบคอลัมน์ Description")

# ================== MEDICAL ANALYTICS ==================
with med_tab:
    st.markdown("## 🩺 Medical Analytics")

    # ---- Disease Analysis by Average Age (ตามจำนวนเคส) ----
    st.subheader("Disease Analysis by Average Age")
    if {"disease_group_mapped","age"}.issubset(df_filtered.columns):
        dis_age = (
            df_filtered.groupby("disease_group_mapped", as_index=False)
            .agg(AverageAge=("age","mean"), Cases=("age","size"))
            .sort_values(["Cases","AverageAge"], ascending=[False, False])
        )
        top_n = st.slider("แสดงสูงสุด (ตามจำนวนเคส)", 5, min(60, len(dis_age)), min(20, len(dis_age)))
        dis_age = dis_age.head(top_n)
        dynamic_height = int(26 * max(5, len(dis_age)) + 80)

        chart = (
            alt.Chart(dis_age, height=dynamic_height)
            .mark_bar()
            .encode(
                x=alt.X("AverageAge:Q", title="Average Age", scale=alt.Scale(nice=True)),
                y=alt.Y("disease_group_mapped:N", sort='-x', title="Disease",
                        axis=alt.Axis(labelLimit=350, labelPadding=6)),
                tooltip=[
                    "disease_group_mapped",
                    alt.Tooltip("AverageAge:Q", format=".1f"),
                    alt.Tooltip("Cases:Q", title="Cases", format=",d")
                ]
            )
            .properties(padding={"left": 10, "right": 10, "top": 10, "bottom": 10})
        )
        st.altair_chart(chart.configure_view(stroke=None), use_container_width=True)
    else:
        st.info("ไม่พบคอลัมน์ที่ต้องใช้ (disease_group_mapped, age)")

    st.markdown("---")
    st.header("🏥🔖 สิทธิการรักษา • โรงพยาบาล • การจ่ายเงิน")

    # ================== PAYER / RIGHTS ANALYSIS ==================
    st.subheader("1) Payer Mix & KPI")
    if "Customer/Vendor Name" in df_filtered.columns:
        payer_agg = (
            df_filtered.groupby("Customer/Vendor Name", as_index=False)
            .agg(
                Revenue=("LineTotal","sum"),
                Profit=("Profit","sum"),
                Cases=("LineTotal","size"),
            )
            .sort_values("Revenue", ascending=False)
        )
        payer_agg["ARPC"] = payer_agg["Revenue"] / payer_agg["Cases"]  # Avg revenue per case
        payer_agg["Margin%"] = np.where(payer_agg["Revenue"]>0, payer_agg["Profit"]/payer_agg["Revenue"]*100, 0.0)

        top_payer_n = st.slider("แสดงสิทธิการรักษาสูงสุด (ตามรายได้)", 5, min(50, len(payer_agg)), min(15, len(payer_agg)))
        payer_show = payer_agg.head(top_payer_n)

        cols = st.columns(2)
        with cols[0]:
            st.altair_chart(
                alt.Chart(payer_show).mark_bar().encode(
                    x=alt.X("Revenue:Q", title="Revenue (฿)"),
                    y=alt.Y("Customer/Vendor Name:N", sort='-x', title="Payer / สิทธิการรักษา",
                            axis=alt.Axis(labelLimit=300, labelPadding=6)),
                    tooltip=[
                        "Customer/Vendor Name",
                        alt.Tooltip("Revenue:Q", format=",.0f"),
                        alt.Tooltip("Cases:Q", format=",d"),
                        alt.Tooltip("ARPC:Q", format=",.0f", title="Avg/Case"),
                        alt.Tooltip("Margin%:Q", format=".1f", title="Margin (%)"),
                    ]
                ),
                use_container_width=True
            )
        with cols[1]:
            st.altair_chart(
                alt.Chart(payer_show).mark_bar().encode(
                    x=alt.X("ARPC:Q", title="Avg Revenue per Case (฿)"),
                    y=alt.Y("Customer/Vendor Name:N", sort='-x', title=None,
                            axis=alt.Axis(labelLimit=300, labelPadding=6)),
                    tooltip=[
                        "Customer/Vendor Name",
                        alt.Tooltip("ARPC:Q", format=",.0f"),
                        alt.Tooltip("Cases:Q", format=",d"),
                        alt.Tooltip("Revenue:Q", format=",.0f"),
                    ]
                ),
                use_container_width=True
            )

        st.dataframe(
            payer_show.assign(
                Revenue=lambda d: d["Revenue"].map(lambda x: f"{x:,.0f}"),
                Profit=lambda d: d["Profit"].map(lambda x: f"{x:,.0f}"),
                ARPC=lambda d: d["ARPC"].map(lambda x: f"{x:,.0f}"),
                **{"Margin%": payer_show["Margin%"].map(lambda x: f"{x:.1f}%")}
            ).rename(columns={"Customer/Vendor Name":"Payer / สิทธิการรักษา"}),
            use_container_width=True
        )
    else:
        st.info("ไม่พบคอลัมน์สิทธิการรักษา (เช่น Customer/Vendor Name)")

    # ================== HOSPITAL x PAYER MATRIX ==================
    st.subheader("2) สิทธิการรักษา × โรงพยาบาล (Cases/Revenue Heatmap)")
    has_cols = {"Customer/Vendor Name","โรงพยาบาล"}.issubset(df_filtered.columns)
    if has_cols:
        metric = st.radio("เลือกตัวชี้วัดสำหรับฮีตแมป", ["Cases","Revenue"], horizontal=True)
        case_tbl = df_filtered.groupby(["โรงพยาบาล","Customer/Vendor Name"]).size().reset_index(name="Cases")
        rev_tbl  = df_filtered.groupby(["โรงพยาบาล","Customer/Vendor Name"])["LineTotal"].sum().reset_index(name="Revenue")
        cross = case_tbl.merge(rev_tbl, on=["โรงพยาบาล","Customer/Vendor Name"], how="outer").fillna(0)

        # rank by chosen metric
        hosp_tot = cross.groupby("โรงพยาบาล")[metric].sum().sort_values(ascending=False)
        payer_tot = cross.groupby("Customer/Vendor Name")[metric].sum().sort_values(ascending=False)

        max_h = max(5, min(30, len(hosp_tot)))
        max_p = max(5, min(30, len(payer_tot)))
        top_h = st.slider("แสดงโรงพยาบาลสูงสุด", 5, max_h, min(12, max_h))
        top_p = st.slider("แสดงสิทธิการรักษาสูงสุด", 5, max_p, min(15, max_p))

        hosp_order = hosp_tot.head(top_h).index.tolist()
        payer_order = payer_tot.head(top_p).index.tolist()
        cross_f = cross[cross["โรงพยาบาล"].isin(hosp_order) & cross["Customer/Vendor Name"].isin(payer_order)].copy()
        cross_f["โรงพยาบาล"] = pd.Categorical(cross_f["โรงพยาบาล"], categories=hosp_order, ordered=True)
        cross_f["Customer/Vendor Name"] = pd.Categorical(cross_f["Customer/Vendor Name"], categories=payer_order, ordered=True)
        heat_height = 32 * len(hosp_order) + 60

        st.altair_chart(
            alt.Chart(cross_f).mark_rect().encode(
                x=alt.X("Customer/Vendor Name:N", title="Payer / สิทธิการรักษา",
                        sort=payer_order, axis=alt.Axis(labelLimit=250, labelPadding=6)),
                y=alt.Y("โรงพยาบาล:N", title="Hospital", sort=hosp_order,
                        axis=alt.Axis(labelLimit=250, labelPadding=6)),
                color=alt.Color(f"{metric}:Q", title=metric, scale=alt.Scale(scheme="blues")),
                tooltip=[
                    "โรงพยาบาล",
                    "Customer/Vendor Name",
                    alt.Tooltip("Cases:Q", title="Cases", format=",d"),
                    alt.Tooltip("Revenue:Q", title="Revenue (฿)", format=",.0f"),
                ],
            ).properties(height=heat_height),
            use_container_width=True,
        )
    else:
        st.info("ต้องมีคอลัมน์ 'โรงพยาบาล' และ 'Customer/Vendor Name'")

    # ================== MONTHLY TRENDS BY PAYER ==================
    st.subheader("3) แนวโน้มรายเดือนตามสิทธิการรักษา (Stacked Area)")
    if {"Customer/Vendor Name","YM","LineTotal"}.issubset(df_filtered.columns):
        # เอาเฉพาะ Top N payers โดยรายได้รวม
        payer_tot = (
            df_filtered.groupby("Customer/Vendor Name")["LineTotal"].sum()
            .sort_values(ascending=False)
        )
        top_p = st.slider("เลือกจำนวน Top Payers ที่แสดง", 3, min(15, len(payer_tot)), min(8, len(payer_tot)))
        top_payers = payer_tot.head(top_p).index.tolist()

        trend = (df_filtered[df_filtered["Customer/Vendor Name"].isin(top_payers)]
                 .groupby(["YM","Customer/Vendor Name"], as_index=False)["LineTotal"].sum())
        # เพื่อให้แกนเวลาเรียงถูก
        trend["YM_ord"] = pd.PeriodIndex(trend["YM"], freq="M").to_timestamp()

        st.altair_chart(
            alt.Chart(trend).mark_area(opacity=0.85).encode(
                x=alt.X("YM_ord:T", title="Month"),
                y=alt.Y("sum(LineTotal):Q", title="Revenue (฿)"),
                color=alt.Color("Customer/Vendor Name:N", title="Payer"),
                tooltip=[
                    alt.Tooltip("YM:N", title="Month"),
                    "Customer/Vendor Name:N",
                    alt.Tooltip("sum(LineTotal):Q", format=",.0f", title="Revenue (฿)")
                ]
            ),
            use_container_width=True
        )
    else:
        st.info("ไม่พบคอลัมน์สำหรับแนวโน้มรายเดือน (YM / Customer/Vendor Name / LineTotal)")

    # ================== PAYMENT METHOD ANALYSIS ==================
    st.subheader("4) การจ่ายเงิน / ช่องทางชำระ")
    if "Payment Method" in df_filtered.columns:
        paym = (df_filtered.groupby("Payment Method", as_index=False)
                .agg(Revenue=("LineTotal","sum"), Cases=("LineTotal","size")))
        paym["ARPC"] = np.where(paym["Cases"]>0, paym["Revenue"]/paym["Cases"], 0)
        paym = paym.sort_values("Revenue", ascending=False)

        cols = st.columns(2)
        with cols[0]:
            st.altair_chart(
                alt.Chart(paym).mark_bar().encode(
                    x=alt.X("Revenue:Q", title="Revenue (฿)"),
                    y=alt.Y("Payment Method:N", sort='-x', title="Payment Method",
                            axis=alt.Axis(labelLimit=300, labelPadding=6)),
                    tooltip=[
                        "Payment Method",
                        alt.Tooltip("Revenue:Q", format=",.0f"),
                        alt.Tooltip("Cases:Q", format=",d"),
                        alt.Tooltip("ARPC:Q", format=",.0f"),
                    ]
                ),
                use_container_width=True
            )
        with cols[1]:
            total_rev = paym["Revenue"].sum()
            paym["Percent"] = np.where(total_rev>0, paym["Revenue"]/total_rev*100, 0.0)
            base = alt.Chart(paym).encode(
                theta=alt.Theta("Revenue:Q"),
                color=alt.Color("Payment Method:N", title="Payment Method"),
                tooltip=[
                    "Payment Method",
                    alt.Tooltip("Revenue:Q", format=",.0f"),
                    alt.Tooltip("Percent:Q", format=".1f", title="Percent (%)"),
                ],
            )
            st.altair_chart(
                base.mark_arc(outerRadius=110) +
                base.mark_text(radius=135, size=12).encode(text=alt.Text("Percent:Q", format=".1f")),
                use_container_width=True
            )

        st.dataframe(
            paym.assign(
                Revenue=lambda d: d["Revenue"].map(lambda x: f"{x:,.0f}"),
                ARPC=lambda d: d["ARPC"].map(lambda x: f"{x:,.0f}"),
                **{"Percent": paym["Percent"].map(lambda x: f"{x:.1f}%")}
            ),
            use_container_width=True
        )
    else:
        st.info("ไม่พบคอลัมน์วิธีชำระเงิน (เช่น Payment Method / วิธีชำระเงิน)")

    # ================== BASIC COLLECTION INSIGHT (OPTIONAL) ==================
    st.subheader("5) สถานะการรับชำระ (ถ้ามีวันจ่าย/ครบกำหนด)")
    # ใช้ได้ถ้ามี Posting Date + (Payment Date หรือ Due Date)
    has_post = "Posting Date" in df_filtered.columns
    has_pay  = "Payment Date" in df_filtered.columns or "Paid Date" in df_filtered.columns
    has_due  = "Due Date" in df_filtered.columns

    if has_post and (has_pay or has_due):
        tmp = df_filtered.copy()
        pay_date_col = "Payment Date" if "Payment Date" in tmp.columns else ("Paid Date" if "Paid Date" in tmp.columns else None)

        # Days to Pay (DTP) และ Days Overdue (ถ้ามี Due Date)
        if pay_date_col:
            tmp["DaysToPay"] = (tmp[pay_date_col] - tmp["Posting Date"]).dt.days
        if has_due:
            tmp["DaysOverdue"] = (tmp.get(pay_date_col, tmp["Due Date"]) - tmp["Due Date"]).dt.days

        # แจกแจงตามสิทธิการรักษา (เฉพาะบรรทัดที่มีค่า)
        show_cols = []
        if "DaysToPay" in tmp.columns:
            dtp_payer = tmp.dropna(subset=["DaysToPay"]).groupby("Customer/Vendor Name", as_index=False)["DaysToPay"].median() \
                           .sort_values("DaysToPay")
            st.altair_chart(
                alt.Chart(dtp_payer).mark_bar().encode(
                    x=alt.X("DaysToPay:Q", title="Median Days to Pay"),
                    y=alt.Y("Customer/Vendor Name:N", sort='-x', title="Payer",
                            axis=alt.Axis(labelLimit=300, labelPadding=6)),
                    tooltip=["Customer/Vendor Name", alt.Tooltip("DaysToPay:Q", format=",d")]
                ),
                use_container_width=True
            )
            show_cols.append("DaysToPay")

        if "DaysOverdue" in tmp.columns:
            dov = tmp.dropna(subset=["DaysOverdue"]).copy()
            # bucket เป็นช่วงๆ เพื่อดูสัดส่วน
            bins = [-9999, -1, 0, 7, 30, 60, 90, 9999]
            labels = ["Early","On time","1–7d","8–30d","31–60d","61–90d",">90d"]
            dov["OverdueBucket"] = pd.cut(dov["DaysOverdue"], bins=bins, labels=labels)
            dist = dov.groupby("OverdueBucket", as_index=False)["LineTotal"].sum()
            total = dist["LineTotal"].sum()
            dist["Percent"] = np.where(total>0, dist["LineTotal"]/total*100, 0.0)

            base = alt.Chart(dist).encode(
                x=alt.X("OverdueBucket:N", title="Overdue Bucket", sort=labels),
                y=alt.Y("Percent:Q", title="Percent (%)"),
                tooltip=[ "OverdueBucket", alt.Tooltip("Percent:Q", format=".1f"),
                          alt.Tooltip("LineTotal:Q", format=",.0f", title="Revenue (฿)") ],
            )
            st.altair_chart(base.mark_bar(), use_container_width=True)
            show_cols.append("DaysOverdue")

        if not show_cols:
            st.info("มีคอลัมน์วันจ่าย/ครบกำหนด แต่ยังคำนวณไม่ได้ (อาจไม่มีค่าในช่วงที่เลือก)")
    else:
        st.caption("ℹ️ ไม่มีข้อมูลวันจ่าย/ครบกำหนดเพียงพอสำหรับวิเคราะห์ DTP/Overdue ในช่วงนี้")

# ================== DATA QUALITY / DEBUG ==================
with st.expander("🔍 Data Quality / Sanity Checks"):
    st.write("จำนวนแถวหลังกรอง:", len(df_filtered))
    st.write("จำนวนสาขา:", df_filtered["Branch"].nunique())
    st.write("ค่าเฉลี่ย LineTotal ต่อแถว:", f"{df_filtered['LineTotal'].mean():,.2f}")
    st.write("ยอดรวมทั้งหมด (หลัง clean + filter):", f"{df_filtered['LineTotal'].sum():,.0f}")

    st.write("ยอดรวมตามสาขา:")
    st.dataframe(
        df_filtered.groupby("Branch", as_index=False)["LineTotal"].sum()
          .sort_values("LineTotal", ascending=False)
          .style.format({"LineTotal":"{:,.0f}"})
    )

    st.write("Top 10 transactions by LineTotal:")
    st.dataframe(
        df_filtered.nlargest(10, "LineTotal")[ ["Posting Date","Branch","Description","LineTotal"] ]
          .style.format({"LineTotal":"{:,.0f}"})
    )

st.markdown("---")
st.caption(
    f"📅 Data: {min_date} → {max_date} | Rows after filter: {len(df_filtered):,} | Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)
