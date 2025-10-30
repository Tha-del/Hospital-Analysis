# ---- Disease Analysis by Average Age (ตามจำนวนเคส) ----
st.subheader("Disease Analysis by Average Age")

if {"disease_group_mapped","age"}.issubset(df_filtered.columns):
    # รวมตามโรค: นับจำนวนเคส + อายุเฉลี่ย
    dis_age = (
        df_filtered.groupby("disease_group_mapped", as_index=False)
        .agg(AverageAge=("age","mean"), Cases=("age","size"))
        .sort_values(["Cases","AverageAge"], ascending=[False, False])
    )

    # ✅ คงไว้เฉพาะตัวเลือก "แสดงสูงสุด (ตามจำนวนเคส)"
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
