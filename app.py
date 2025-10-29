import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime

# ---------------- Page config ----------------
st.set_page_config(layout="wide", page_title="Business & Medical Dashboard")
st.title("📊 Business and Medical Analytics Dashboard")

# ---------------- Load Parquet -----------------
try:
    df = pd.read_parquet("final_test_data_20250529.parquet")
except FileNotFoundError:
    st.error("❌ ไม่พบไฟล์: final_test_data_20250529.parquet (กรุณาวางไฟล์ Parquet ไว้ในโฟลเดอร์เดียวกับสคริปต์)")
    st.stop()
except Exception as e:
    st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์ Parquet: {e}")
    st.stop()

# ------------- Data Cleaning / Preparation -------------
# Numeric
df["avg_cost"]  = pd.to_numeric(df.get("avg_cost", 0),  errors="coerce").fillna(0)
df["LineTotal"] = pd.to_numeric(df.get("LineTotal", 0), errors="coerce").fillna(0)
df["Quantity"]  = pd.to_numeric(df.get("Quantity", 1),  errors="coerce").fillna(1)

# Mappings
disease_mapping = {
    "กล้ามเนื้อเคล็ด": "Muscle Strain",
    "โรคทางเดินปัสสาวะ": "Urinary Tract Disease",
    "ปัจจัยที่มีผลต่อสถานะสุขภาพ": "Factors Affecting Health Status",
    "ความผิดปกติจากทางคลินิกและห้องปฏิบัติการ": "Abnormalities from Clinical",
    "โรคทางเดินอาหาร": "Gastrointestinal Disease",
    "URI": "Upper Respiratory Infection (URI)",
    "การติดเชื้อไวรัส": "Viral Infection",
    "การบาดเจ็บ การเป็นพิษ และอุบัติเหตุ": "Injury, Poisoning, and Accidents",
}
gender_mapping = {"M": "Male", "F": "Female", "W": "Female", "ชาย": "Male", "หญิง": "Female"}

if "group_disease" not in df.columns:
    st.error("ไม่พบคอลัมน์ group_disease")
    st.stop()
df["disease_group_mapped"] = df["group_disease"].map(disease_mapping)

if "เพศ คนไข้" in df.columns:
    df["gender_mapped"] = df["เพศ คนไข้"].map(gender_mapping).fillna("Other")
else:
    df["gender_mapped"] = "Other"

# Age group (optional)
if "age" in df.columns:
    bins = [0, 17, 35, 55, 100]
    labels = ["0–17", "18–35", "36–55", "56+"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels)

# Dates
if "Posting Date" not in df.columns:
    st.error("ไม่พบคอลัมน์ Posting Date")
    st.stop()
df["Posting Date"] = pd.to_datetime(df["Posting Date"], errors="coerce")

# ---------------- Sidebar filters ----------------
st.sidebar.header("⚙️ Filters")

profit_formula = st.sidebar.selectbox(
    "สูตรคำนวณกำไร",
    (
        "Per-Unit Cost: LineTotal - (avg_cost × Quantity)",
        "Current: LineTotal - avg_cost",
        "Fixed 40% Margin: LineTotal × 0.40",
    ),
    help="เลือกสูตรกำไรที่จะใช้ในทุกกราฟ/ตาราง"
)

# Profit calc
if profit_formula.startswith("Per-Unit"):
    df["Profit"] = df["LineTotal"] - (df["avg_cost"] * df["Quantity"])
elif profit_formula.startswith("Current"):
    df["Profit"] = df["LineTotal"] - df["avg_cost"]
else:
    df["Profit"] = df["LineTotal"] * 0.40

# Date range + Branch dropdown
min_date = df["Posting Date"].min().date()
max_date = df["Posting Date"].max().date()
date_range = st.sidebar.date_input("Select date range", (min_date, max_date),
                                   min_value=min_date, max_value=max_date)

branches = sorted(df["Branch"].dropna().unique().tolist())
branch_options = ["ทุกสาขา"] + branches
branch_choice = st.sidebar.selectbox("Branch (เลื่อนลงเลือก)", branch_options, index=0)

# Filter
start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
selected_branches = branches if branch_choice == "ทุกสาขา" else [branch_choice]

df_filtered = df[
    (df["Posting Date"] >= start_date) &
    (df["Posting Date"] <= end_date) &
    (df["Branch"].isin(selected_branches))
].copy()

df_filtered.dropna(subset=["Branch", "disease_group_mapped"], inplace=True)
if df_filtered.empty:
    st.warning("ไม่พบข้อมูลตามตัวกรองที่เลือก")
    st.stop()

# Derived fields
df_filtered["YM"]   = df_filtered["Posting Date"].dt.to_period("M").astype(str)
df_filtered["Year"] = df_filtered["Posting Date"].dt.year

st.caption(f"🧮 Using profit formula: {profit_formula}")

st.markdown("---")
tab1, tab2 = st.tabs(["📈 Business Analytics", "🩺 Medical Analytics"])

# ======================================================
# =================== BUSINESS TAB =====================
# ======================================================
with tab1:
    st.header("📈 Business Analytics")

    # ---------- Top 5 Branches by Revenue and Profit ----------
    st.subheader("Top 5 Branches by Revenue and Profit")
    branch_perf = (
        df_filtered.groupby("Branch")[["LineTotal", "Profit"]]
        .sum()
        .sort_values("LineTotal", ascending=False)
        .head(5)
        .reset_index()
    )
    chart = (
        alt.Chart(branch_perf)
        .transform_fold(["LineTotal", "Profit"], as_=["Metric", "Value"])
        .mark_bar()
        .encode(
            x=alt.X("Branch:N", title="Branch"),
            y=alt.Y("Value:Q", title="Amount (Baht)"),
            color=alt.Color("Metric:N", title="Metric", scale=alt.Scale(scheme="tableau10")),
            tooltip=["Branch:N", "Metric:N", alt.Tooltip("Value:Q", format=",.0f")],
        )
    )
    st.altair_chart(chart, use_container_width=True)

    # ---------- Product Revenue Contribution (pie + %) ----------
    st.subheader("Product Revenue Contribution")
    desc_col = next((c for c in ["Description", "Dscription", "dscription", "description"]
                     if c in df_filtered.columns), None)
    if desc_col:
        prod = (
            df_filtered.groupby(desc_col)["LineTotal"]
            .sum()
            .reset_index()
            .sort_values("LineTotal", ascending=False)
            .head(10)
        )
        total_sum = prod["LineTotal"].sum()
        prod["Percent"] = prod["LineTotal"] / total_sum * 100

        pie_base = alt.Chart(prod).encode(
            theta=alt.Theta("LineTotal:Q", stack=True),
            color=alt.Color(f"{desc_col}:N", title="Product/Service"),
            tooltip=[
                alt.Tooltip(f"{desc_col}:N", title="Product"),
                alt.Tooltip("LineTotal:Q", title="Revenue", format=",.0f"),
                alt.Tooltip("Percent:Q", title="Percent (%)", format=".1f"),
            ],
        )
        pie = pie_base.mark_arc(outerRadius=120)
        text = pie_base.mark_text(radius=145, size=12, color="black").encode(
            text=alt.Text("Percent:Q", format=".1f")
        )
        st.altair_chart(pie + text, use_container_width=True)

    # ---------- Branch Monthly Performance & Strategy ----------
    st.markdown("### 📅 Branch Monthly Performance & Strategy")

    # Monthly revenue by branch
    monthly_branch = (
        df_filtered.groupby(["Branch", "YM"])["LineTotal"]
        .sum()
        .reset_index(name="Revenue")
        .sort_values(["Branch", "YM"])
    )

    # Median per branch (reference)
    branch_median = monthly_branch.groupby("Branch")["Revenue"].median().rename("BranchMedian")

    # Prev month revenue for MoM
    monthly_branch = monthly_branch.sort_values(["Branch", "YM"])
    monthly_branch["PrevRevenue"] = monthly_branch.groupby("Branch")["Revenue"].shift(1)

    # Strategy table base
    strat_df = monthly_branch.merge(branch_median, on="Branch", how="left")
    for c in ["Revenue", "BranchMedian", "PrevRevenue"]:
        strat_df[c] = pd.to_numeric(strat_df[c], errors="coerce")

    strat_df["PerfIndex"] = strat_df["Revenue"] / strat_df["BranchMedian"]
    strat_df["MoM"] = (strat_df["Revenue"] - strat_df["PrevRevenue"]) / strat_df["PrevRevenue"]

    # Top product & mix concentration
    if desc_col:
        by_prod = df_filtered.groupby(["Branch", "YM", desc_col])["LineTotal"].sum().reset_index()
        total_by_month = by_prod.groupby(["Branch", "YM"])["LineTotal"].sum().rename("Total").reset_index()
        top_by_month = (
            by_prod.sort_values(["Branch", "YM", "LineTotal"], ascending=[True, True, False])
            .groupby(["Branch", "YM"]).first().reset_index()
            .rename(columns={desc_col: "TopProduct", "LineTotal": "TopProductRevenue"})
        )
        mix = top_by_month.merge(total_by_month, on=["Branch", "YM"])
        mix["TopShare"] = mix["TopProductRevenue"] / mix["Total"]
        strat_df = strat_df.merge(mix[["Branch", "YM", "TopProduct", "TopShare"]], on=["Branch", "YM"], how="left")
    else:
        strat_df["TopProduct"] = "Key Service"
        strat_df["TopShare"] = pd.NA

    # Strategy rules
    def recommend(row):
        pi  = row["PerfIndex"]
        mom = row["MoM"]
        ts  = row["TopShare"]
        tp  = row.get("TopProduct", "Key Service")

        if pd.isna(pi):
            return "ตรวจสอบข้อมูล"

        high_con = pd.notna(ts) and ts >= 0.60
        recovering = pd.notna(mom) and mom > 0.08
        slipping   = pd.notna(mom) and mom < -0.10

        if pi < 0.7:
            return (f"Turnaround | โปรแรง/แพ็กเกจ, outbound ลูกค้าเก่า"
                    f"{' | ลดการพึ่งพา '+tp if high_con else ''}")
        if pi < 0.9:
            if recovering: return "Recovering | เพิ่ม budget โฆษณาเฉพาะแคมเปญที่เวิร์ก"
            if slipping:   return "At-Risk | ทบทวนราคา, cross-sell, follow-up ด้วย SMS/Line"
            return f"Below Median | ดันเรือธง '{tp}', ทำดีลองค์กร"
        if pi < 1.1:
            if slipping:   return "Stable but Slipping | A/B test โฆษณา, referral program"
            return "Stable | รักษา mix และเพิ่ม add-on margin สูง"
        if pi < 1.2:
            return ("Good"
                    f"{' but Concentrated | กระจายพอร์ตจาก '+tp if high_con else ' | ขยาย budget ช่วง peak'}")
        return ("Star"
                f"{' but Concentrated | ลด dependence จาก '+tp if high_con else ' | เพิ่ม capacity/PR'}")

    strat_df["Strategy"] = strat_df.apply(recommend, axis=1)

    # ------------ Monthly View (ไม่มี Heatmap) ------------
    st.markdown("#### 📈 Monthly View")
    view_mode = st.radio(
        "โหมดการแสดงผล",
        ["Highlight 1 สาขา", "Top-N Line"],
        horizontal=True,
    )

    line_base = monthly_branch.copy()
    line_base["Revenue"] = pd.to_numeric(line_base["Revenue"], errors="coerce")
    ordered_months = sorted(line_base["YM"].unique())
    line_base["YM"] = pd.Categorical(line_base["YM"], categories=ordered_months, ordered=True)

    if view_mode == "Highlight 1 สาขา":
        target = st.selectbox("เลือกสาขาที่ต้องการไฮไลต์", sorted(df_filtered["Branch"].unique()))
        line = (
            alt.Chart(line_base)
            .mark_line(point=True)
            .encode(
                x=alt.X("YM:N", title="Month"),
                y=alt.Y("Revenue:Q", title="Revenue (฿)"),
                color=alt.Color("Branch:N", legend=None),
                opacity=alt.condition(alt.datum.Branch == target, alt.value(1), alt.value(0.12)),
                tooltip=["YM", "Branch", alt.Tooltip("Revenue:Q", format=",.0f")],
            )
        )
        st.altair_chart(line.properties(height=420).interactive(), use_container_width=True)

    else:  # Top-N Line
        N = st.slider("จำนวนสาขา (Top-N)", 3, min(20, line_base["Branch"].nunique()), 8)
        topN = (
            df_filtered.groupby("Branch")["LineTotal"].sum()
            .sort_values(ascending=False).head(N).index.tolist()
        )
        d_top = line_base[line_base["Branch"].isin(topN)]
        line = (
            alt.Chart(d_top)
            .mark_line(point=True)
            .encode(
                x=alt.X("YM:N", title="Month"),
                y=alt.Y("Revenue:Q", title="Revenue (฿)"),
                color=alt.Color("Branch:N", title="Branch"),
                tooltip=["YM", "Branch", alt.Tooltip("Revenue:Q", format=",.0f")],
            )
        )
        st.altair_chart(line.properties(height=420).interactive(), use_container_width=True)

    # ------------ Strategy table (6 months latest) ------------
    latest_ym = sorted(strat_df["YM"].unique())[-6:]
    show = strat_df[strat_df["YM"].isin(latest_ym)][
        ["YM", "Branch", "Revenue", "BranchMedian", "PerfIndex", "MoM", "TopShare", "TopProduct", "Strategy"]
    ].sort_values(["YM", "PerfIndex"]).copy()

    for c in ["Revenue", "BranchMedian", "PerfIndex", "MoM", "TopShare"]:
        show[c] = pd.to_numeric(show[c], errors="coerce")

    def row_style(row):
        pi = row["PerfIndex"]
        if pd.isna(pi): return [''] * len(row)
        if pi < 0.7:  return ['background-color:#2a1215;color:#ffd6d9'] * len(row)
        if pi < 0.9:  return ['background-color:#2b2610;color:#fff1b8'] * len(row)
        if pi < 1.1:  return ['background-color:#1f1f1f;color:#d9d9d9'] * len(row)
        if pi < 1.2:  return ['background-color:#112a2a;color:#b7eb8f'] * len(row)
        return ['background-color:#0f2a16;color:#95de64'] * len(row)

    st.dataframe(
        show.style
            .apply(row_style, axis=1)
            .format({
                "Revenue": "{:,.0f}".format,
                "BranchMedian": "{:,.0f}".format,
                "PerfIndex": "{:.2f}".format,
                "MoM": (lambda x: "" if pd.isna(x) else f"{x*100:.1f}%"),
                "TopShare": (lambda x: "" if pd.isna(x) else f"{x*100:.1f}%")
            }),
        use_container_width=True
    )
    st.caption("สีตาราง: 🔴 PI<0.7 | 🟡 0.7–0.89 | ⚪ 0.9–1.09 | 🟢 1.1–1.19 | 🟢🟩 ≥1.2")

    # ------------------- Annual Totals -------------------
    st.markdown("### 📦 Annual Totals")

    annual_overall = df_filtered.groupby("Year")["LineTotal"].sum().reset_index(name="TotalRevenue")
    annual_overall["TotalRevenue"] = pd.to_numeric(annual_overall["TotalRevenue"], errors="coerce")
    st.write("**ยอดรวมทั้งปี (รวมทุกสาขา)**")
    st.dataframe(annual_overall.style.format({"TotalRevenue": "{:,.0f}".format}), use_container_width=True)

    annual_branch = df_filtered.groupby(["Year", "Branch"])["LineTotal"].sum().reset_index(name="TotalRevenue")
    pivot_year_branch = annual_branch.pivot(index="Branch", columns="Year", values="TotalRevenue").fillna(0)
    pivot_year_branch = pivot_year_branch.apply(pd.to_numeric, errors="coerce")
    st.write("**ยอดรวมทั้งปี แยกตามสาขา**")
    st.dataframe(pivot_year_branch.style.format("{:,.0f}".format), use_container_width=True)

# ======================================================
# =================== MEDICAL TAB ======================
# ======================================================
# =========================================================
# 🩺 MEDICAL ANALYTICS (ใหม่)
# =========================================================
# =========================================================
# 🩺 MEDICAL ANALYTICS (อัปเดตใหม่)
# =========================================================
with tab2:
    st.header("🩺 Medical Analytics")

    # ========= 1️⃣ Disease Analysis (อายุเฉลี่ยต่อกลุ่มโรค) =========
    st.subheader("🦠 Disease Analysis by Average Patient Age")
    if "group_disease" in df_filtered.columns and "age" in df_filtered.columns:
        disease_age = (
            df_filtered.groupby("group_disease")["age"]
            .mean()
            .reset_index()
            .sort_values("age", ascending=False)
        )
        chart_disease = (
            alt.Chart(disease_age)
            .mark_bar(color="#ff7f0e")
            .encode(
                x=alt.X("age:Q", title="Average Age (Years)"),
                y=alt.Y("group_disease:N", sort='-x', title="Disease Group"),
                tooltip=[
                    alt.Tooltip("group_disease:N", title="Disease"),
                    alt.Tooltip("age:Q", format=".1f", title="Average Age"),
                ],
            )
            .properties(height=400)
        )
        st.altair_chart(chart_disease, use_container_width=True)
    else:
        st.info("ไม่พบคอลัมน์ 'group_disease' หรือ 'age' ในข้อมูล")

    st.markdown("---")

    # ========= 2️⃣ Service (บริการ / หัตถการ / ยา) =========
    st.subheader("💊 Top 10 Services or Procedures by Revenue")
    desc_col = next(
        (c for c in ["Description", "Dscription", "dscription", "description"] if c in df_filtered.columns),
        None,
    )
    if desc_col:
        service_rev = (
            df_filtered.groupby(desc_col)["LineTotal"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
            .rename(columns={desc_col: "Service"})
        )
        chart_service = (
            alt.Chart(service_rev)
            .mark_bar(color="#1f77b4")
            .encode(
                x=alt.X("LineTotal:Q", title="Total Revenue (฿)"),
                y=alt.Y("Service:N", sort="-x", title="Service / Procedure"),
                tooltip=[
                    alt.Tooltip("Service:N", title="Service"),
                    alt.Tooltip("LineTotal:Q", format=",.0f", title="Revenue (฿)"),
                ],
            )
            .properties(height=400)
        )
        st.altair_chart(chart_service, use_container_width=True)
    else:
        st.info("ไม่พบคอลัมน์ Description หรือ Dscription ในข้อมูล")

    st.markdown("---")

    # ========= 3️⃣ Gender (เพศผู้ป่วย) =========
    st.subheader("🚻 Patient Distribution by Gender")
    if "เพศ คนไข้" in df_filtered.columns:
        gender_map = {"M": "Male", "F": "Female", "ชาย": "Male", "หญิง": "Female", "W": "Female"}
        df_filtered["gender_mapped"] = df_filtered["เพศ คนไข้"].map(gender_map).fillna("Other")

        gender_count = (
            df_filtered.groupby("gender_mapped")["LineTotal"]
            .sum()
            .reset_index()
            .rename(columns={"LineTotal": "Revenue"})
        )
        total = gender_count["Revenue"].sum()
        gender_count["Percent"] = gender_count["Revenue"] / total * 100

        pie = (
            alt.Chart(gender_count)
            .mark_arc(innerRadius=40)
            .encode(
                theta=alt.Theta("Revenue:Q", title="Revenue"),
                color=alt.Color("gender_mapped:N", title="Gender", scale=alt.Scale(scheme="pastel1")),
                tooltip=[
                    alt.Tooltip("gender_mapped:N", title="Gender"),
                    alt.Tooltip("Revenue:Q", format=",.0f", title="Revenue (฿)"),
                    alt.Tooltip("Percent:Q", format=".1f", title="Percent (%)"),
                ],
            )
        )
        text = pie.mark_text(radius=120, size=13, color="black").encode(text=alt.Text("Percent:Q", format=".1f"))
        st.altair_chart(pie + text, use_container_width=True)
    else:
        st.info("ไม่พบคอลัมน์ 'เพศ คนไข้' ในข้อมูล")

    st.markdown("---")

    # ========= 4️⃣ Age (กลุ่มอายุ) =========
    st.subheader("👶🧓 Age Distribution")
    if "age" in df_filtered.columns:
        bins = [0, 17, 35, 55, 100]
        labels = ["0–17", "18–35", "36–55", "56+"]
        df_filtered["age_group"] = pd.cut(df_filtered["age"], bins=bins, labels=labels)

        age_rev = (
            df_filtered.groupby("age_group")["LineTotal"]
            .sum()
            .reset_index()
            .rename(columns={"LineTotal": "Revenue"})
        )

        chart_age = (
            alt.Chart(age_rev)
            .mark_bar(color="#2ca02c")
            .encode(
                x=alt.X("age_group:N", title="Age Group"),
                y=alt.Y("Revenue:Q", title="Total Revenue (฿)"),
                tooltip=[
                    alt.Tooltip("age_group:N", title="Age Group"),
                    alt.Tooltip("Revenue:Q", format=",.0f", title="Revenue (฿)"),
                ],
            )
            .properties(height=400)
        )
        st.altair_chart(chart_age, use_container_width=True)
    else:
        st.info("ไม่พบคอลัมน์ 'age' ในข้อมูล")
# =========================================================
# 🏥 HOSPITAL & PAYER ANALYTICS
# =========================================================

st.markdown("---")
st.header("🏥 Hospital & Payer Analysis")

# ---------- 1️⃣ Branch Performance ----------
st.subheader("🏆 Hospital Revenue Performance (Top 10)")
if "โรงพยาบาล" in df_filtered.columns:
    hosp_perf = (
        df_filtered.groupby("โรงพยาบาล")["LineTotal"]
        .sum()
        .reset_index()
        .sort_values("LineTotal", ascending=False)
        .head(10)
    )

    chart_hosp = (
        alt.Chart(hosp_perf)
        .mark_bar(color="#1f77b4")
        .encode(
            x=alt.X("LineTotal:Q", title="Total Revenue (฿)"),
            y=alt.Y("โรงพยาบาล:N", sort='-x', title="Hospital / Branch"),
            tooltip=[
                alt.Tooltip("โรงพยาบาล:N", title="Hospital"),
                alt.Tooltip("LineTotal:Q", format=",.0f", title="Revenue (฿)"),
            ],
        )
        .properties(height=400)
    )
    st.altair_chart(chart_hosp, use_container_width=True)
else:
    st.info("ไม่พบคอลัมน์ 'โรงพยาบาล' ในข้อมูล")

# ---------- 2️⃣ Cross Analysis ----------
st.markdown("---")
st.subheader("🔄 Cross Analysis: Payer Type × Hospital")

if {"Customer/Vendor Name", "โรงพยาบาล"}.issubset(df_filtered.columns):
    cross = (
        df_filtered.groupby(["Customer/Vendor Name", "โรงพยาบาล"])["LineTotal"]
        .sum()
        .reset_index(name="Revenue")
    )

    # แสดงเฉพาะ Top 10 โรงพยาบาล เพื่อไม่ให้กราฟแน่นเกินไป
    top_hosp = (
        cross.groupby("โรงพยาบาล")["Revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .index.tolist()
    )
    cross_filtered = cross[cross["โรงพยาบาล"].isin(top_hosp)]

    chart_cross = (
        alt.Chart(cross_filtered)
        .mark_rect()
        .encode(
            x=alt.X("โรงพยาบาล:N", title="Hospital / Branch"),
            y=alt.Y("Customer/Vendor Name:N", title="Payer Type"),
            color=alt.Color(
                "Revenue:Q",
                scale=alt.Scale(scheme="tealblues"),
                title="Revenue (฿)"
            ),
            tooltip=[
                alt.Tooltip("โรงพยาบาล:N", title="Hospital"),
                alt.Tooltip("Customer/Vendor Name:N", title="Payer Type"),
                alt.Tooltip("Revenue:Q", format=",.0f", title="Revenue (฿)"),
            ],
        )
        .properties(height=450)
    )
    st.altair_chart(chart_cross, use_container_width=True)
else:
    st.info("ไม่พบคอลัมน์ 'Customer/Vendor Name' หรือ 'โรงพยาบาล' ในข้อมูล")


