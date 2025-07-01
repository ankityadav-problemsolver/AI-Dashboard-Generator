
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from dotenv import load_dotenv
import google.generativeai as genai
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from fpdf import FPDF # type: ignore
import base64
from io import BytesIO

# === Load environment variables ===
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

# === Team Login ===
def login():
    st.sidebar.header("🔐 Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if username == "admin" and password == "pass123":
            st.session_state.logged_in = True
        else:
            st.sidebar.error("Invalid credentials")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
    st.stop()

# === App Config ===
st.set_page_config(page_title="📊 Smart AI Dashboard", layout="wide")
st.markdown("<h1 style='text-align:center;'>📈 Dynamic AI Dashboard Generator</h1>", unsafe_allow_html=True)

# === File Upload ===
uploaded_file = st.file_uploader("📁 Upload Excel or CSV", type=["csv", "xlsx"])
df = None

if uploaded_file:
    ext = uploaded_file.name.split(".")[-1]
    if ext == "csv":
        df = pd.read_csv(uploaded_file)
    else:
        sheet = st.selectbox("📑 Choose a Sheet", pd.ExcelFile(uploaded_file).sheet_names)
        df = pd.read_excel(uploaded_file, sheet_name=sheet)

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    st.success("✅ File Uploaded and Processed")

    # === Preview Data ===
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # === Filters ===
    if 'brand' in df.columns:
        brands = df['brand'].dropna().unique()
        selected = st.multiselect("🔘 Filter by Brand", brands, default=brands)
        df = df[df['brand'].isin(selected)]

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        date_range = st.date_input("📆 Select Date Range", [df['date'].min(), df['date'].max()])
        df = df[(df['date'] >= date_range[0]) & (df['date'] <= date_range[1])]

    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # === KPI Cards ===
    st.subheader("📌 KPIs")
    k1, k2, k3 = st.columns(3)
    k1.metric("📄 Total Records", len(df))
    if num_cols:
        k2.metric(f"📊 Mean {num_cols[0]}", round(df[num_cols[0]].mean(), 2))
        k3.metric(f"🔻 Min {num_cols[0]}", round(df[num_cols[0]].min(), 2))

    # === Pie Chart ===
    if cat_cols:
        pie_data = df[cat_cols[0]].value_counts().head(10)
        fig_pie = px.pie(values=pie_data.values, names=pie_data.index, title=f"Top {cat_cols[0]} Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)

    # === Bar Chart ===
    if len(cat_cols) > 0 and len(num_cols) > 0:
        avg_df = df.groupby(cat_cols[0])[num_cols[0]].mean().reset_index()
        fig_bar = px.bar(avg_df, x=cat_cols[0], y=num_cols[0], title=f"{num_cols[0]} by {cat_cols[0]}")
        st.plotly_chart(fig_bar, use_container_width=True)

    # === Forecasting with Advanced ML ===
    if 'date' in df.columns and len(num_cols) > 0:
        st.subheader("🔮 Advanced Forecasting")
        forecast_col = st.selectbox("📈 Forecast Column", num_cols)
        df = df.sort_values("date")
        df['days'] = (df['date'] - df['date'].min()).dt.days
        X = df[['days']]
        y = df[forecast_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model_rf = RandomForestRegressor()
        model_rf.fit(X_train, y_train)
        df['forecast'] = model_rf.predict(X)

        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=df['date'], y=df[forecast_col], name="Actual", mode='lines+markers'))
        fig_forecast.add_trace(go.Scatter(x=df['date'], y=df['forecast'], name="Forecast", mode='lines'))
        fig_forecast.update_layout(title=f"{forecast_col} Forecast with Random Forest")
        st.plotly_chart(fig_forecast, use_container_width=True)

        st.info(f"📊 R² Score: {r2_score(y_test, model_rf.predict(X_test)):.2f}")

    # === Export Options ===
    st.subheader("📤 Export Data")
    if st.button("📥 Export CSV"):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="dashboard_data.csv">Download CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

    if st.button("📄 Export PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for col in df.columns:
            pdf.cell(200, 10, txt=f"{col}: {df[col].iloc[0]}", ln=True)
        buffer = BytesIO()
        pdf.output(buffer)
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode()
        st.markdown(f'<a href="data:application/pdf;base64,{b64}" download="report.pdf">Download PDF</a>', unsafe_allow_html=True)

    # === Gemini ChatBot ===
    st.subheader("💬 Chat with your Data")
    user_query = st.text_area("🧠 Ask Gemini something about your dataset")

    if st.button("🤖 Get Answer"):
        preview = df.head(10).to_markdown(index=False)
        describe = df.describe().to_markdown()
        prompt = f"""
        You're a data analyst. Answer the question based on this dataset.
        
        Sample:
        {preview}
        
        Summary:
        {describe}
        
        Question:
        {user_query}
        """
        try:
            with st.spinner("Analyzing with Gemini..."):
                result = model.generate_content(prompt)
                st.markdown("### 🧠 Gemini's Answer")
                st.markdown(result.text)
        except Exception as e:
            st.error(f"Gemini Error: {e}")


