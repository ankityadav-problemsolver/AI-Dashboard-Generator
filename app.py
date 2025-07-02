import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
# from fpdf import FPDF
import base64
from io import BytesIO
import os
import google.generativeai as genai

# === Load environment variables once ===
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

# === Caching for faster reloads ===
@st.cache_data
def load_data(file, ext, sheet=None):
    if ext == "csv":
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file, sheet_name=sheet)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

@st.cache_resource
def get_gemini_response(prompt):
    return model.generate_content(prompt).text

# === Login System ===
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

# === Streamlit Config ===
st.set_page_config(page_title="📊 AI Dashboard", layout="wide")
st.markdown("<h1 style='text-align:center;'>📈 Smart AI Dashboard</h1>", unsafe_allow_html=True)

# === File Upload ===
uploaded_file = st.file_uploader("📁 Upload CSV or Excel", type=["csv", "xlsx"])
df = None

if uploaded_file:
    ext = uploaded_file.name.split(".")[-1]
    if ext == "xlsx":
        sheet_names = pd.ExcelFile(uploaded_file).sheet_names
        selected_sheet = st.selectbox("📑 Choose Sheet", sheet_names)
        df = load_data(uploaded_file, ext, selected_sheet)
    else:
        df = load_data(uploaded_file, ext)

    st.success("✅ Data Loaded")
    st.dataframe(df.head(10), use_container_width=True)

    # === Filter Section ===
    with st.expander("📂 Filters"):
        if "brand" in df.columns:
            brands = df["brand"].dropna().unique()
            selected_brands = st.multiselect("Filter by Brand", brands, default=brands)
            df = df[df["brand"].isin(selected_brands)]

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors='coerce')
            df = df.dropna(subset=["date"])
            date_range = st.date_input("Date Range", [df["date"].min(), df["date"].max()])
            df = df[(df["date"] >= date_range[0]) & (df["date"] <= date_range[1])]

    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # === KPIs ===
    st.subheader("📌 Key Metrics")
    k1, k2, k3 = st.columns(3)
    k1.metric("🔢 Total Rows", len(df))
    if num_cols:
        k2.metric(f"📊 Avg {num_cols[0]}", round(df[num_cols[0]].mean(), 2))
        k3.metric(f"📉 Min {num_cols[0]}", round(df[num_cols[0]].min(), 2))

    # === Charts Section ===
    st.subheader("📊 Visual Insights")
    if cat_cols:
        pie_data = df[cat_cols[0]].value_counts().nlargest(10)
        fig_pie = px.pie(values=pie_data.values, names=pie_data.index, hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

    if num_cols and cat_cols:
        bar_df = df.groupby(cat_cols[0])[num_cols[0]].mean().reset_index()
        fig_bar = px.bar(bar_df, x=cat_cols[0], y=num_cols[0])
        st.plotly_chart(fig_bar, use_container_width=True)

    if len(num_cols) >= 2:
        xcol = st.selectbox("X-Axis", num_cols)
        ycol = st.selectbox("Y-Axis", num_cols, index=1)
        fig_scatter = px.scatter(df, x=xcol, y=ycol, color=cat_cols[0] if cat_cols else None)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # === Forecasting ===
    if "date" in df.columns and num_cols:
        st.subheader("🔮 Forecasting")
        target = st.selectbox("Forecast Column", num_cols)
        df = df.sort_values("date")
        df["days"] = (df["date"] - df["date"].min()).dt.days
        X = df[["days"]]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        rf = RandomForestRegressor()
        rf.fit(X_train, y_train)
        df["prediction"] = rf.predict(X)

        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=df["date"], y=df[target], mode="lines+markers", name="Actual"))
        fig_forecast.add_trace(go.Scatter(x=df["date"], y=df["prediction"], mode="lines", name="Forecast"))
        fig_forecast.update_layout(title=f"{target} Forecast (Random Forest)")
        st.plotly_chart(fig_forecast, use_container_width=True)

        st.caption(f"✅ R² Score: {r2_score(y_test, rf.predict(X_test)):.2f}")

    # === Gemini Chat ===
    st.subheader("💬 Gemini AI Assistant")
    question = st.text_area("Ask your data question to Gemini")
    if st.button("🤖 Get Insight"):
        short_data = df.head(10).to_markdown(index=False)
        summary = df.describe().round(2).to_markdown()
        prompt = f"""
        You're a data analyst. Analyze this data and answer the question below.

        Sample Data:
        {short_data}

        Summary:
        {summary}

        Question:
        {question}
        """
        with st.spinner("Gemini is analyzing..."):
            try:
                result = get_gemini_response(prompt)
                st.markdown("### 🔍 Gemini's Response")
                st.markdown(result)
            except Exception as e:
                st.error(f"Error from Gemini: {e}")

    # === Export Section ===
    st.subheader("📤 Export Data")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬇️ Export as CSV"):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            st.markdown(f'<a href="data:file/csv;base64,{b64}" download="dashboard.csv">Download CSV</a>', unsafe_allow_html=True)
    with col2:
        if st.button("⬇️ Export as PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=10)
            for col in df.columns:
                pdf.cell(200, 8, txt=f"{col}: {df[col].iloc[0]}", ln=True)
            buffer = BytesIO()
            pdf.output(buffer)
            buffer.seek(0)
            b64 = base64.b64encode(buffer.read()).decode()
            st.markdown(f'<a href="data:application/pdf;base64,{b64}" download="report.pdf">Download PDF</a>', unsafe_allow_html=True)
