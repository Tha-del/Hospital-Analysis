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

    # unify description column
    desc_alias = [c for c in ["Description","Dscription","dscription","description"] if c in df.columns]
    if desc_alias:
        if desc_alias[0] != "Description":
            df.rename(columns={desc_alias[0]: "Description"}, inplace=True)

    # types
    if "Posting Date" in df.columns:
        df["Posting Date"] = pd.to_datetime(df["Posting Date"], errors="coerce")

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

    # map gender / age group / disease
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

    # ---------- strict de-duplication ----------
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

# เปิด/ปิด strict de-dup ใน sidebar เพื่อไล่ต้นตอได้
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

    # ---- Top 5 Branches by Revenue and Profit ----
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

    # ---- Product Revenue Contribution (Top 10) ----
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
        st.altair_chart(base.mark_arc(outerRadius=120) +
                        base.mark_text(radius=145, size=12).encode(text=alt.Text("Percent:Q", format=".1f")),
                        use_container_width=True)
    else:
        st.info("ไม่พบคอลัมน์ Description")

    # ---- Branch Monthly + Strategy ----
    st.markdown("### 📅 Branch Monthly Performance & Strategy")
    monthly_branch = (
        df_filtered.groupby(["Branch","YM"])["LineTotal"]
          .sum().reset_index(name="Revenue")
          .sort_values(["Branch","YM"])
    )
    branch_median = monthly_branch.groupby("Branch")["Revenue"].median().rename("BranchMedian")
    monthly_branch = monthly_branch.merge(branch_median, on="Branch", how="left")
    monthly_branch["PrevRevenue"] = monthly_branch.groupby("Branch")["Revenue"].shift(1)
    monthly_branch["PerfIndex"]   = monthly_branch["Revenue"]/monthly_branch["BranchMedian"]
    monthly_branch["MoM"]         = (monthly_branch["Revenue"]-monthly_branch["PrevRevenue"])/monthly_branch["PrevRevenue"]

    # top product share ต่อเดือน-สาขา
    if "Description" in df_filtered.columns:
        by_prod = df_filtered.groupby(["Branch","YM","Description"])["LineTotal"].sum().reset_index()
        total_mb = by_prod.groupby(["Branch","YM"]) ["LineTotal"].sum().rename("Total").reset_index()
        top_mb   = (by_prod.sort_values(["Branch","YM","LineTotal"], ascending=[True,True,False])
                          .groupby(["Branch","YM"]).first().reset_index()
                          .rename(columns={"Description":"TopProduct","LineTotal":"TopProductRevenue"}))
        mix = top_mb.merge(total_mb, on=["Branch","YM"])
        mix["TopShare"] = mix["TopProductRevenue"] / mix["Total"]
        monthly_branch = monthly_branch.merge(mix[["Branch","YM","TopProduct","TopShare"]],
                                              on=["Branch","YM"], how="left")
    else:
        monthly_branch["TopProduct"] = "Key Service"
        monthly_branch["TopShare"]   = np.nan

    def recommend(row):
        pi = row["PerfIndex"]; mom=row["MoM"]; ts=row["TopShare"]; tp=row.get("TopProduct","Key")
        high_con = pd.notna(ts) and ts>=0.60
        recovering = pd.notna(mom) and mom>0.08
        slipping   = pd.notna(mom) and mom<-0.10
        if pd.isna(pi): return "ตรวจสอบข้อมูล"
        if pi<0.7: return f"Turnaround | โปรแรง/แพ็กเกจ{' | ลดพึ่งพา '+tp if high_con else ''}"
        if pi<0.9:
            if recovering: return "Recovering | เพิ่มงบแคมเปญที่เวิร์ก"
            if slipping:   return "At-Risk | ทบทวนราคา + cross-sell + follow-up"
            return f"Below Median | ดันเรือธง '{tp}'"
        if pi<1.1: return "Stable" if not slipping else "Stable but Slipping | A/B test + referral"
        if pi<1.2: return "Good" + (" but Concentrated | กระจายพอร์ตจาก "+tp if high_con else " | ขยาย budget ช่วง peak")
        return "Star" + (" but Concentrated | ลด dependence จาก "+tp if high_con else " | เพิ่ม capacity/PR")

    monthly_branch["Strategy"] = monthly_branch.apply(recommend, axis=1)

    branches_for_line = st.multiselect(
        "เลือกสาขาสำหรับกราฟแนวโน้มรายเดือน",
        sorted(df_filtered["Branch"].unique().tolist()),
        default=sorted(df_filtered["Branch"].unique().tolist())[:5]
    )
    if branches_for_line:
        line_data = monthly_branch[monthly_branch["Branch"].isin(branches_for_line)]
        st.altair_chart(
            alt.Chart(line_data).mark_line(point=True).encode(
                x=alt.X("YM:N", title="Month"),
                y=alt.Y("Revenue:Q", title="Revenue (฿)"),
                color="Branch:N",
                tooltip=["YM","Branch",alt.Tooltip("Revenue:Q",format=",.0f")]
            ).interactive(),
            use_container_width=True
        )

    latest_ym = sorted(monthly_branch["YM"].unique())[-6:]
    show = monthly_branch[monthly_branch["YM"].isin(latest_ym)][
        ["YM","Branch","Revenue","BranchMedian","PerfIndex","MoM","TopShare","TopProduct","Strategy"]
    ].sort_values(["YM","PerfIndex"])

    def row_style(r):
        pi=r["PerfIndex"]
        if pd.isna(pi): return ['']*len(r)
        if pi<0.7:  return ['background-color:#2a1215;color:#ffd6d9']*len(r)
        if pi<0.9:  return ['background-color:#2b2610;color:#fff1b8']*len(r)
        if pi<1.1:  return ['background-color:#1f1f1f;color:#d9d9d9']*len(r)
        if pi<1.2:  return ['background-color:#112a2a;color:#b7eb8f']*len(r)
        return ['background-color:#0f2a16;color:#95de64']*len(r)

    st.dataframe(
        show.style.apply(row_style, axis=1).format({
            "Revenue":"{:,.0f}".format,
            "BranchMedian":"{:,.0f}".format,
            "PerfIndex":"{:,.2f}".format,
            "MoM": lambda x: "" if pd.isna(x) else f"{x*100:.1f}%",
            "TopShare": lambda x: "" if pd.isna(x) else f"{x*100:.1f}%"
        }),
        use_container_width=True
    )
    st.caption("สี: 🔴 PI<0.7 | 🟡 0.7–0.89 | ⚪ 0.9–1.09 | 🟢 1.1–1.19 | 🟢🟩 ≥1.2")

    # ---- Annual Totals ----
    st.markdown("### 📦 Annual Totals")
    annual_overall = (
        df_filtered.assign(Year=lambda x: x["Posting Date"].dt.year)
          .groupby("Year", as_index=False)["LineTotal"].sum()
          .rename(columns={"LineTotal":"TotalRevenue"})
    )
    st.write("**ยอดรวมทั้งปี (รวมทุกสาขา)**")
    st.dataframe(annual_overall.style.format({"TotalRevenue":"{:,.0f}"}), use_container_width=True)

    annual_branch = (
        df_filtered.assign(Year=lambda x: x["Posting Date"].dt.year)
          .groupby(["Branch","Year"], as_index=False)["LineTotal"].sum()
          .rename(columns={"LineTotal":"TotalRevenue"})
    )
    pivot_year_branch = annual_branch.pivot(index="Branch", columns="Year", values="TotalRevenue").fillna(0)
    st.write("**ยอดรวมทั้งปี แยกตามสาขา**")
    st.dataframe(pivot_year_branch.style.format("{:,.0f}"), use_container_width=True)

# ================== MEDICAL ANALYTICS ==================
with med_tab:
    st.markdown("## 🩺 Medical Analytics")

    # ---- Disease Analysis by Average Age (NO min-case filter) ----
    st.subheader("Disease Analysis by Average Age")
    if {"disease_group_mapped","age"}.issubset(df_filtered.columns):
        dis_age = (df_filtered.groupby("disease_group_mapped", as_index=False)
                   .agg(AverageAge=("age","mean"), Cases=("age","size")))

        # แสดงสูงสุดตามจำนวนเคสอย่างเดียว (เอาตัวกรองขั้นต่ำออกแล้ว)
        top_n = st.slider("แสดงสูงสุด (ตามจำนวนเคส)", 5, 60, 20)
        dis_age = (dis_age.sort_values(["Cases","AverageAge"], ascending=[False, False])
                   .head(top_n))

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

    # ---- Top Services / Procedures by Revenue (Top 10) ----
    st.subheader("Top Services / Procedures by Revenue (Top 10)")
    if "Description" in df_filtered.columns:
        top_srv = (df_filtered.groupby("Description")["LineTotal"]
                   .sum().reset_index()
                   .sort_values("LineTotal", ascending=False).head(10))
        st.altair_chart(
            alt.Chart(top_srv).mark_bar().encode(
                x=alt.X("LineTotal:Q", title="Revenue (฿)"),
                y=alt.Y("Description:N", sort='-x', title="Service / Procedure",
                        axis=alt.Axis(labelLimit=350, labelPadding=6)),
                tooltip=[alt.Tooltip("LineTotal:Q", format=",.0f"), "Description:N"]
            ),
            use_container_width=True
        )

    st.subheader("Patient Distribution by Gender (Revenue Share)")
    if "gender_mapped" in df_filtered.columns:
        by_gender = df_filtered.groupby("gender_mapped")["LineTotal"].sum().reset_index()
        by_gender["Percent"] = by_gender["LineTotal"] / by_gender["LineTotal"].sum() * 100 if by_gender["LineTotal"].sum()>0 else 0
        base_g = alt.Chart(by_gender).encode(
            theta=alt.Theta("LineTotal:Q"),
            color="gender_mapped:N",
            tooltip=["gender_mapped", alt.Tooltip("LineTotal:Q", format=",.0f"),
                     alt.Tooltip("Percent:Q", format=".1f")]
        )
        st.altair_chart(base_g.mark_arc(outerRadius=110) +
                        base_g.mark_text(radius=135, size=12).encode(text=alt.Text("Percent:Q", format=".1f")),
                        use_container_width=True)

    st.subheader("Revenue by Age Group")
    if "age_group" in df_filtered.columns:
        age_rev = df_filtered.groupby("age_group")["LineTotal"].sum().reset_index()
        st.altair_chart(
            alt.Chart(age_rev).mark_bar().encode(
                x=alt.X("age_group:N", title="Age Group"),
                y=alt.Y("LineTotal:Q", title="Revenue (฿)"),
                tooltip=[alt.Tooltip("LineTotal:Q", format=",.0f"), "age_group:N"]
            ),
            use_container_width=True
        )

    # ================== SIMPLIFIED CROSS ANALYSIS (BY CASES) ==================
    st.subheader("สิทธิ์การรักษา × โรงพยาบาล (ตามจำนวนเคส)")
    if {"Customer/Vendor Name","โรงพยาบาล"}.issubset(df_filtered.columns):
        # Count cases per Hospital × Payer, keep Revenue for tooltip only
        case_tbl = df_filtered.groupby(["โรงพยาบาล","Customer/Vendor Name"])\
                              .size().reset_index(name="Cases")
        rev_tbl  = df_filtered.groupby(["โรงพยาบาล","Customer/Vendor Name"])\
                              ["LineTotal"].sum().reset_index(name="Revenue")
        cross = case_tbl.merge(rev_tbl, on=["โรงพยาบาล","Customer/Vendor Name"], how="left")

        # Totals to determine top entities by cases
        hosp_tot = cross.groupby("โรงพยาบาล")["Cases"].sum().sort_values(ascending=False)
        payer_tot = cross.groupby("Customer/Vendor Name")["Cases"].sum().sort_values(ascending=False)

        # Sliders for top-N (by cases)
        max_h = max(5, min(30, len(hosp_tot)))
        max_p = max(5, min(30, len(payer_tot)))
        top_h = st.slider("แสดงโรงพยาบาลสูงสุด (ตามจำนวนเคส)", 5, max_h, min(12, max_h))
        top_p = st.slider("แสดงสิทธิ์การรักษาสูงสุด (ตามจำนวนเคส)", 5, max_p, min(15, max_p))

        hosp_order = hosp_tot.head(top_h).index.tolist()
        payer_order = payer_tot.head(top_p).index.tolist()

        cross_f = cross[cross["โรงพยาบาล"].isin(hosp_order) & cross["Customer/Vendor Name"].isin(payer_order)].copy()
        # ensure sort orders
        cross_f["โรงพยาบาล"] = pd.Categorical(cross_f["โรงพยาบาล"], categories=hosp_order, ordered=True)
        cross_f["Customer/Vendor Name"] = pd.Categorical(cross_f["Customer/Vendor Name"], categories=payer_order, ordered=True)

        heat_height = 32 * len(hosp_order) + 60

        st.altair_chart(
            alt.Chart(cross_f).mark_rect().encode(
                x=alt.X("Customer/Vendor Name:N", title="Payer / สิทธิ์การรักษา",
                        sort=payer_order,
                        axis=alt.Axis(labelLimit=250, labelPadding=6)),
                y=alt.Y("โรงพยาบาล:N", title="Hospital", sort=hosp_order,
                        axis=alt.Axis(labelLimit=250, labelPadding=6)),
                color=alt.Color("Cases:Q", title="Cases", scale=alt.Scale(scheme="blues")),
                tooltip=[
                    "โรงพยาบาล",
                    "Customer/Vendor Name",
                    alt.Tooltip("Cases:Q", title="Cases", format=",d"),
                    alt.Tooltip("Revenue:Q", title="Revenue (฿)", format=",.0f"),
                ],
            ).properties(height=heat_height),
            use_container_width=True,
        )

# ================== DATA QUALITY / DEBUG ==================
with st.expander("🔍 Data Quality / Sanity Checks"):
    st.write("จำนวนแถวหลังกรอง:", len(df_filtered))
    st.write("จำนวนสาขา:", df_filtered["Branch"].nunique())
    st.write("ค่าเฉลี่ย LineTotal ต่อแถว:", f"{df_filtered['LineTotal'].mean():,.2f}")
    st.write("ยอดรวมทั้งหมด (หลัง clean + filter):", f"{df_filtered['LineTotal'].sum():,.0f}")

    # ดูยอดต่อสาขา (ช่วยตรวจว่ามีสาขาไหนกระโดด)
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
