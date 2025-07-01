import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from dotenv import load_dotenv
import os
import google.generativeai as genai

# Load .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

st.set_page_config(page_title="📊 AI-Powered Smart Dashboard", layout="wide")
st.title("📊 Universal Dashboard Generator with Gemini AI")

# File uploader
uploaded_file = st.file_uploader("📁 Upload Excel or CSV", type=["xlsx", "csv"])

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1]
    if file_type == "csv":
        df = pd.read_csv(uploaded_file)
    else:
        xls = pd.ExcelFile(uploaded_file)
        sheets = xls.sheet_names
        selected_sheet = st.selectbox("Select Sheet", sheets)
        df = pd.read_excel(xls, sheet_name=selected_sheet)

    st.success("✅ Data Loaded Successfully")
    st.subheader("📄 Preview Data")
    st.dataframe(df.head(10), use_container_width=True)

    # Filters
    if 'brand' in df.columns:
        brands = df['brand'].unique()
        selected_brands = st.multiselect("Filter by Brand", brands, default=brands)
        df = df[df['brand'].isin(selected_brands)]

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        start_date, end_date = st.date_input("Filter by Date Range", [df['date'].min(), df['date'].max()])
        df = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]

    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # KPI Cards
    st.subheader("📌 Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("🔢 Total Records", len(df))
    if 'range_km' in df.columns:
        col2.metric("🔋 Avg Range", round(df['range_km'].mean(), 2))
    if 'efficiency_wh_per_km' in df.columns:
        col3.metric("⚡ Efficiency (Wh/km)", round(df['efficiency_wh_per_km'].mean(), 2))

    # Visuals
    st.subheader("📈 Auto Insights - Charts")

    if numeric_cols:
        col = st.selectbox("🔹 Distribution of:", numeric_cols)
        fig1 = px.histogram(df, x=col, title=f"Distribution of {col}")
        st.plotly_chart(fig1, use_container_width=True)

    if len(numeric_cols) >= 2:
        xcol = st.selectbox("🟦 X-Axis", numeric_cols, key='xcol')
        ycol = st.selectbox("🟥 Y-Axis", numeric_cols, key='ycol')
        color_by = st.selectbox("🎨 Color By (Optional)", [None] + categorical_cols)
        fig2 = px.scatter(df, x=xcol, y=ycol, color=color_by, title=f"{ycol} vs {xcol}")
        st.plotly_chart(fig2, use_container_width=True)

    if len(numeric_cols) >= 3:
        fig3 = px.scatter_3d(df, x=numeric_cols[0], y=numeric_cols[1], z=numeric_cols[2],
                             color=categorical_cols[0] if categorical_cols else None,
                             title="3D Analysis of Key Variables")
        st.plotly_chart(fig3, use_container_width=True)

    # Forecasting
    if 'date' in df.columns and 'range_km' in df.columns:
        st.subheader("🔮 Forecasting - EV Range Over Time")
        df = df.sort_values('date')
        df['days_since'] = (df['date'] - df['date'].min()).dt.days
        X = df[['days_since']]
        y = df['range_km']
        model_lr = LinearRegression()
        model_lr.fit(X, y)
        df['predicted_range'] = model_lr.predict(X)

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=df['date'], y=df['range_km'], name="Actual", mode="lines+markers"))
        fig4.add_trace(go.Scatter(x=df['date'], y=df['predicted_range'], name="Predicted", mode="lines"))
        fig4.update_layout(title="🔋 Predicted Range vs Time")
        st.plotly_chart(fig4, use_container_width=True)

    # Gemini AI Insights
    st.subheader("🧠 Gemini AI-Powered Insights")

    prompt = f"""
    You're an AI data analyst. Analyze the uploaded dataset.

    Here's a sample of the data (first 20 rows):
    {df.head(20).to_markdown(index=False)}

    Here's a statistical summary:
    {df.describe().round(2).to_markdown()}

    Return:
    - Key insights or trends
    - Outliers or anomalies
    - Recommendations
    - Predictive modeling suggestions
    - Business takeaways
    """

    if st.button("🧠 Generate Insights with Gemini"):
        with st.spinner("Analyzing with Gemini..."):
            try:
                response = model.generate_content(prompt)
                st.markdown("### 💡 AI Insights")
                st.markdown(response.text)
            except Exception as e:
                st.error(f"Error: {e}")
