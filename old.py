import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from dotenv import load_dotenv
import os
import google.generativeai as genai

# === ENV ===
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# === Gemini Config ===
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

# === Streamlit Page Setup ===
st.set_page_config(page_title="📊 AI Dashboard Generator", layout="wide")
st.markdown("<h1 style='text-align:center;'>📊 AI-Enhanced Business Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

# === File Upload ===
uploaded_file = st.file_uploader("📁 Upload Excel or CSV File", type=["csv", "xlsx"])

if uploaded_file:
    file_ext = uploaded_file.name.split(".")[-1]
    if file_ext == "csv":
        df = pd.read_csv(uploaded_file)
    else:
        xls = pd.ExcelFile(uploaded_file)
        sheet = st.selectbox("📄 Select Sheet", xls.sheet_names)
        df = pd.read_excel(xls, sheet_name=sheet)

    # Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    st.success("✅ Data Loaded Successfully!")
    st.markdown("### 🔍 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # === Filters ===
    if 'brand' in df.columns:
        brands = df['brand'].dropna().unique()
        selected = st.multiselect("🔘 Filter by Brand", brands, default=brands)
        df = df[df['brand'].isin(selected)]

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.dropna(subset=['date'], inplace=True)
        min_d, max_d = df['date'].min(), df['date'].max()
        start, end = st.date_input("📆 Date Range", [min_d, max_d])
        df = df[(df['date'] >= pd.to_datetime(start)) & (df['date'] <= pd.to_datetime(end))]

    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # === KPI Cards ===
    st.markdown("## 📌 Key Metrics")
    k1, k2, k3 = st.columns(3)
    k1.metric("📄 Records", len(df))
    if num_cols:
        k2.metric(f"📊 Avg {num_cols[0]}", round(df[num_cols[0]].mean(), 2))
        k3.metric(f"🔻 Min {num_cols[0]}", round(df[num_cols[0]].min(), 2))

    st.markdown("---")

    # === PIE CHART ===
    if cat_cols:
        st.markdown("## 🥧 Category Distribution (Pie Chart)")
        col_p1, col_p2 = st.columns([2, 1])

        pie_data = df[cat_cols[0]].value_counts()
        pie_top = pie_data[:10]
        pie_top["Other"] = pie_data[10:].sum()

        fig_pie = px.pie(pie_top.reset_index(), names='index', values=cat_cols[0], hole=0.4)
        fig_pie.update_traces(textinfo='percent+label')

        col_p1.plotly_chart(fig_pie, use_container_width=True)
        col_p2.markdown(
            f"""
            <div style='background-color:#f0f2f6;padding:10px;border-radius:5px;'>
            <b>🧠 Insight:</b> Top 10 categories of <code>{cat_cols[0]}</code> shown.<br>
            Others grouped under <b>Other</b> to maintain clarity.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # === BAR CHART ===
    if cat_cols and num_cols:
        st.markdown("## 📊 Average Metric by Category (Bar Chart)")
        col_b1, col_b2 = st.columns([2, 1])

        bar_data = df.groupby(cat_cols[0])[num_cols[0]].mean().reset_index()
        fig_bar = px.bar(bar_data, x=cat_cols[0], y=num_cols[0], title=f"Average {num_cols[0]} by {cat_cols[0]}")

        col_b1.plotly_chart(fig_bar, use_container_width=True)
        col_b2.markdown(
            f"""
            <div style='background-color:#f0f2f6;padding:10px;border-radius:5px;'>
            <b>🧠 Insight:</b> Visualizes how <code>{num_cols[0]}</code> differs across <code>{cat_cols[0]}</code> values.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # === SCATTER CHART ===
    if len(num_cols) >= 2:
        st.markdown("## 📈 Correlation Explorer (Scatter Plot)")
        x_axis = st.selectbox("🔷 X-Axis", num_cols, index=0)
        y_axis = st.selectbox("🔶 Y-Axis", num_cols, index=1)
        color = st.selectbox("🎨 Color By", [None] + cat_cols)

        fig_scatter = px.scatter(df, x=x_axis, y=y_axis, color=color, title=f"{y_axis} vs {x_axis}")
        st.plotly_chart(fig_scatter, use_container_width=True)

    # === FORECASTING CHART ===
    if 'date' in df.columns and num_cols:
        st.markdown("## 🔮 Forecasting Trend Over Time")
        forecast_col = st.selectbox("📈 Column to Forecast", num_cols)
        df = df.sort_values("date")
        df['days'] = (df['date'] - df['date'].min()).dt.days

        model_lr = LinearRegression()
        model_lr.fit(df[['days']], df[forecast_col])
        df['forecast'] = model_lr.predict(df[['days']])

        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=df['date'], y=df[forecast_col], name='Actual', mode='lines+markers'))
        fig_forecast.add_trace(go.Scatter(x=df['date'], y=df['forecast'], name='Forecast', mode='lines'))
        fig_forecast.update_layout(title=f"📈 Forecast of {forecast_col} Over Time")
        st.plotly_chart(fig_forecast, use_container_width=True)

    # === GEMINI INSIGHTS ===
    st.markdown("---")
    st.markdown("## 🤖 AI-Powered Business Insights")

    sample = df.head(20).to_markdown(index=False)
    summary = df.describe().round(2).to_markdown()

    prompt = f"""
    You are a data analyst. Analyze the dataset below.

    ### Sample Data
    {sample}

    ### Summary
    {summary}

    Provide:
    - Key insights
    - Outliers
    - Recommendations
    - Business actions
    - AI/ML opportunities
    """

    if st.button("🧠 Generate Gemini Insights"):
        with st.spinner("Generating insights..."):
            try:
                response = model.generate_content(prompt)
                st.markdown("### 💡 Insights")
                st.markdown(response.text)
            except Exception as e:
                st.error(f"Error from Gemini: {e}")
