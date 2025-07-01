import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from dotenv import load_dotenv
import google.generativeai as genai
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, roc_curve, auc, confusion_matrix
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.callbacks import EarlyStopping
from fpdf import FPDF
import base64
from io import BytesIO
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# === ENV & Gemini Setup ===
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel("gemini-2.0-flash")

st.set_page_config(page_title="AI Dashboard 🧠", layout="wide")
st.markdown("# 🔍 Enhanced AI Dashboard with Forecasting & Classification")

# === Model Selector ===
mode = st.sidebar.selectbox("Choose Mode", ["Forecasting", "Classification"])

# === File Import ===
uploaded = st.file_uploader("Upload CSV/Excel", type=["csv","xlsx"])
if not uploaded:
    st.stop()

df = pd.read_csv(uploaded) if uploaded.name.endswith("csv") else pd.read_excel(uploaded)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
st.write("### Data Preview", df.head())

# === Email Config Notice ===
if mode == "Forecasting" and (not EMAIL_USER or not EMAIL_PASS):
    st.warning("⚠️ Set EMAIL_USER & EMAIL_PASS in `.env` to enable email sending.")

# === Forecasting Workflow ===
if mode == "Forecasting":
    # Date detection
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    num_cols = df.select_dtypes("number").columns.tolist()
    metric = st.selectbox("Forecast Metric", num_cols)

    horizon = st.number_input("Forecast Days", value=30, min_value=1)
    model_choice = st.selectbox("Algo", ["Prophet", "Linear", "RandomForest", "LSTM"])

    df_ts = df[['date', metric]].rename(columns={'date': 'ds', metric: 'y'})

    if model_choice == "Prophet":
        m = Prophet()
        m.fit(df_ts)
        future = m.make_future_dataframe(periods=horizon)
        fcst = m.predict(future)

    elif model_choice == "Linear":
        df_ts['days'] = (df_ts['ds'] - df_ts['ds'].min()).dt.days
        lr = LinearRegression().fit(df_ts[['days']], df_ts['y'])
        fcst = df_ts.copy()
        fcst['yhat'] = lr.predict(df_ts[['days']])

    elif model_choice == "RandomForest":
        df_ts['days'] = (df_ts['ds'] - df_ts['ds'].min()).dt.days
        rf = RandomForestRegressor().fit(df_ts[['days']], df_ts['y'])
        fcst = df_ts.copy()
        fcst['yhat'] = rf.predict(df_ts[['days']])

    else:
        ts = df_ts['y'].values
        gen = TimeseriesGenerator(ts, ts, length=10, batch_size=1)
        model_l = Sequential([LSTM(50, activation='relu', input_shape=(10,1)), Dense(1)])
        model_l.compile(optimizer='adam', loss='mse')
        model_l.fit(gen, epochs=10, callbacks=[EarlyStopping(patience=2)])
        preds = model_l.predict(gen)
        fcst = df_ts.iloc[10:].copy()
        fcst['yhat'] = preds.flatten()

    # Plot results
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fcst['ds'], y=fcst['y'], name="Actual"))
    fig.add_trace(go.Scatter(x=fcst['ds'], y=fcst['yhat'], name="Forecast"))
    st.plotly_chart(fig, use_container_width=True)

    # Display R² if test data present
    if 'yhat' in fcst.columns:
        r2 = r2_score(fcst['y'], fcst['yhat'])
        st.metric("R² Score", f"{r2:.2f}")

# === Classification Workflow ===
else:
    cat_cols = df.select_dtypes('object').columns.tolist()
    target = st.selectbox("Target Label", cat_cols)
    features = st.multiselect("Features", [c for c in df.columns if c != target])
    if len(features) >= 1:
        X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.3)
        clf = RandomForestClassifier().fit(X_train, y_train)
        preds = clf.predict(X_test)
        st.metric("Accuracy", accuracy_score(y_test, preds))

        # Confusion
        cm = confusion_matrix(y_test, preds)
        st.write("Confusion Matrix", cm)

        # ROC-AUC
        if len(np.unique(y_test)) == 2:  # binary
            probs = clf.predict_proba(X_test)[:,1]
            fpr, tpr, _ = roc_curve(y_test, probs, pos_label=clf.classes_[1])
            st.write("ROC-AUC Curve", auc(fpr, tpr))
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines'))
            fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash')))
            st.plotly_chart(fig_roc, use_container_width=True)

# === Gemini Email Insights ===
if st.button("Generate & Send Report"):
    text = genai.generate_content(f"Provide insights on this dataset: {df.describe().to_markdown()}").text
    st.write("💡 AI Insights:", text)

    # Build PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(190, 8, text)
    buf = BytesIO()
    pdf.output(buf)
    buf.seek(0)

    # Send email
    if EMAIL_USER and EMAIL_PASS:
        msg = MIMEMultipart()
        msg['Subject'] = "Automated Dashboard Report"
        msg['From'], msg['To'] = EMAIL_USER, EMAIL_USER
        msg.attach(MIMEText(text, 'plain'))
        msg.attach(MIMEText(buf.getvalue().decode('latin-1'), 'base64'))
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as s:
            s.login(EMAIL_USER, EMAIL_PASS)
            s.send_message(msg)
        st.success("📧 Report emailed successfully!")

st.markdown("---")
st.info("Deploy on Streamlit Cloud by connecting your repo or use Hugging Face Spaces with `streamlit run app.py`")
