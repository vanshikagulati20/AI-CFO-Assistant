import streamlit as st
import pandas as pd
import pdfplumber
import os
import re
import logging
import json
import requests
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIG & SETUP ---
st.set_page_config(layout="wide", page_title="AI CFO Dashboard", page_icon="🤖")

# --- PASTE YOUR API KEY HERE ---
GEMINI_API_KEY = "AIzaSyC2d8qv3wVAxqELLGwL_nu7wLglzylkf78"
# -----------------------------

SAMPLE_PDF_PATH = "Zomato_Annual_Report_2023-24.pdf"
PREPROCESSED_SAMPLE_PATH = "zomato_chunks.json"


# --- Forecasting Function ---
def generate_forecast(df, years_to_forecast=3):
    forecast_data = []
    last_year = df['Year'].max()
    for metric in ['Revenue', 'Profit']:
        coeffs = np.polyfit(df['Year'], df[metric], 1)
        poly = np.poly1d(coeffs)
        future_years = range(last_year + 1, last_year + 1 + years_to_forecast)
        for year in future_years:
            forecast_value = poly(year)
            forecast_data.append({'Year': year, 'Value': forecast_value, 'Metric': metric, 'Type': 'Forecast'})
    hist_df = df.melt(id_vars='Year', value_vars=['Revenue', 'Profit'], var_name='Metric', value_name='Value')
    hist_df['Type'] = 'Historical'
    forecast_df = pd.DataFrame(forecast_data)
    return pd.concat([hist_df, forecast_df])

# --- Dashboard Building Function ---
def build_dashboard(data):
    """Builds the insightful dashboard with the original UI style."""
    st.header("Financial Health Dashboard", divider='blue')

    history = data.get('historical_performance', [])
    if not history:
        st.warning("No historical performance data was extracted or provided.")
        return
        
    if len(history) < 2:
        st.info("Displaying data for a single period. Trend analysis and forecasting require at least two data points.")
        st.subheader("Key Performance Indicators")
        latest_year_data = history[0]
        col1, col2 = st.columns(2)
        col1.metric(label=f"Total Revenue ({latest_year_data.get('year')})", value=f"₹{latest_year_data.get('revenue_crore', 0):,.0f} Cr")
        col2.metric(label=f"Profit After Tax ({latest_year_data.get('year')})", value=f"₹{latest_year_data.get('profit_crore', 0):,.0f} Cr")
        return

    hist_df = pd.DataFrame(history).rename(columns={'year': 'Year', 'revenue_crore': 'Revenue', 'profit_crore': 'Profit'}).sort_values('Year').reset_index(drop=True)
    latest_year_data = hist_df.iloc[-1]
    previous_year_data = hist_df.iloc[-2]

    st.subheader("Key Performance Indicators (YoY)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label=f"Total Revenue ({latest_year_data['Year']})", value=f"₹{latest_year_data['Revenue']:,.0f} Cr", delta=f"{latest_year_data['Revenue'] - previous_year_data['Revenue']:,.0f} Cr")
    with col2:
        st.metric(label=f"Profit After Tax ({latest_year_data['Year']})", value=f"₹{latest_year_data['Profit']:,.0f} Cr", delta=f"{latest_year_data['Profit'] - previous_year_data['Profit']:,.0f} Cr")
    with col3:
        npm_latest = (latest_year_data['Profit'] / latest_year_data['Revenue']) * 100 if latest_year_data['Revenue'] else 0
        npm_previous = (previous_year_data['Profit'] / previous_year_data['Revenue']) * 100 if previous_year_data['Revenue'] else 0
        st.metric(label=f"Net Profit Margin ({latest_year_data['Year']})", value=f"{npm_latest:.2f}%", delta=f"{npm_latest - npm_previous:.2f}%")

    st.markdown("---")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Revenue Breakdown")
        revenue_breakdown = data.get('revenue_breakdown_crore', [])
        if revenue_breakdown:
            breakdown_df = pd.DataFrame(revenue_breakdown)
            fig = px.pie(breakdown_df, names='segment', values='revenue', title=f"Revenue Mix ({latest_year_data['Year']})", hole=0.4)
            fig.update_traces(textposition='inside', textinfo='percent+label', marker=dict(colors=px.colors.sequential.Tealgrn))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Revenue breakdown by segment was not found.")
    with col2:
        st.subheader("Historical Performance")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=hist_df['Year'], y=hist_df['Revenue'], name='Revenue', marker_color='#1f77b4'))
        fig.add_trace(go.Bar(x=hist_df['Year'], y=hist_df['Profit'], name='Profit', marker_color='#ff7f0e'))
        fig.update_layout(title_text='Historical Revenue vs. Profit', barmode='group', xaxis_title='Year', yaxis_title='Amount (in ₹ Crore)')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("3-Year Financial Forecast")
        forecast_df = generate_forecast(hist_df)
        fig = px.line(forecast_df, x='Year', y='Value', color='Metric', line_dash='Type', title='Historical Data & Future Projections', labels={'Value': 'Amount (in ₹ Crore)'})
        fig.update_traces(line=dict(width=3))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Projected Revenue Share")
        future_rev_df = forecast_df[(forecast_df['Metric'] == 'Revenue') & (forecast_df['Type'] == 'Forecast')]
        if not future_rev_df.empty:
            fig = px.pie(future_rev_df, names='Year', values='Value', title='Share of Forecasted Revenue by Year', hole=0.4)
            fig.update_traces(textposition='inside', textinfo='percent+label', marker=dict(colors=px.colors.sequential.Blues))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Forecast could not be generated.")

    st.markdown("---")
    
    st.subheader("Key Risks & Opportunities")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Identified Risks:**");
        for risk in data.get('future_risks', ["No specific risks identified."]): st.warning(f"🚨 {risk}")
    with col2:
        st.write("**Identified Opportunities:**");
        for opp in data.get('future_opportunities', ["No specific opportunities identified."]): st.success(f"💡 {opp}")

# --- Master Prompt Function (Unchanged) ---
def create_master_prompt(processed_data):
    system_prompt = (
        "You are a financial data extraction expert. Your task is to analyze text chunks from a financial report and extract key metrics. "
        "You must return the data in a valid JSON format. Do not provide any explanation, only the JSON object. "
        "Find data for at least the two most recent fiscal years."
    )
    user_prompt = f"""
    Based on the document chunks, extract financial data. All monetary values should be in Indian Crore (₹), returned as numbers.

    JSON Schema to follow:
    {{
      "historical_performance": [
        {{ "year": number, "revenue_crore": number, "profit_crore": number }},
        {{ "year": number, "revenue_crore": number, "profit_crore": number }}
      ],
      "revenue_breakdown_crore": [
         {{ "segment": "Name of Business Segment (e.g., Food Delivery)", "revenue": number }},
         {{ "segment": "Another Segment", "revenue": number }}
      ],
      "future_risks": [ "A potential future risk mentioned in the report." ],
      "future_opportunities": [ "A potential future opportunity mentioned in the report." ]
    }}

    DOCUMENT CHUNKS:
    {json.dumps(processed_data.get('content_chunks', []), indent=2)}
    """
    return system_prompt, user_prompt

# --- API call and Data loading functions (Unchanged) ---
def run_gemini_extraction(system_prompt, user_prompt, api_key):
    if not api_key or "YOUR_API_KEY_HERE" in api_key:
        st.error("API Key is missing."); return None
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
    payload = {"systemInstruction": {"parts": [{"text": system_prompt}]}, "contents": [{"parts": [{"text": user_prompt}]}], "generationConfig": {"responseMimeType": "application/json"}}
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        result = response.json()
        json_string = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', "")
        if not json_string:
            st.error("The model returned an empty response."); st.json(result); return None
        return json.loads(json_string)
    except json.JSONDecodeError:
        st.error("Failed to decode JSON from the model's response."); st.code(json_string, language="text"); return None
    except Exception as e:
        st.error(f"API Request Failed: {e}"); return None

# --- Data Processing Functions for Uploads (Unchanged) ---
@st.cache_data(show_spinner=False)
def process_uploaded_pdf(uploaded_file):
    """Processes a newly uploaded PDF file."""
    st.write("Extracting text from PDF..."); full_text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        total_pages = len(pdf.pages)
        progress_bar = st.progress(0, text=f"Processing {total_pages} pages...")
        for i, page in enumerate(pdf.pages):
            full_text += f"\n\n--- Page {i+1} ---\n\n"
            page_text = page.extract_text();
            if page_text: full_text += page_text
            progress_bar.progress((i + 1) / total_pages, text=f"Processed page {i+1}/{total_pages}")
    st.write("Shredding document into chunks...")
    text = re.sub(r'\s+', ' ', full_text).strip()
    chunks = [text[i:i+2000] for i in range(0, len(text), 1800)]
    return {"source_type": "PDF", "content_chunks": chunks}

@st.cache_data(show_spinner=False)
def process_structured_file(uploaded_file):
    """Processes a newly uploaded CSV or Excel file."""
    st.write("Loading structured data...")
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        # For this version, we will create a dictionary that the dashboard can use
        latest_year = pd.to_datetime(df['Date']).dt.year.max() if 'Date' in df.columns else "N/A"
        revenue = df['Income'].sum() if 'Income' in df.columns else 0
        profit = revenue - (df['Expense'].sum() if 'Expense' in df.columns else 0)
        
        # We manually construct the dictionary that the build_dashboard function expects
        return {
            "historical_performance": [{"year": latest_year, "revenue_crore": revenue, "profit_crore": profit}],
            "revenue_breakdown_crore": [], 
            "future_risks": ["Manual analysis needed for raw transaction data."], 
            "future_opportunities": ["Scaling customer base and improving margins are key opportunities."]
        }
    except Exception as e:
        st.error(f"An error occurred processing the file: {e}"); return None

@st.cache_resource
def get_or_create_sample_chunks(pdf_path, json_path):
    if not os.path.exists(json_path):
        st.warning(f"Performing one-time processing of '{pdf_path}'...")
        if not os.path.exists(pdf_path):
            st.error(f"Sample PDF '{pdf_path}' not found."); st.stop()
        with st.spinner("Processing..."):
            full_text = "";
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text: full_text += page_text
            text = re.sub(r'\s+', ' ', full_text).strip()
            chunks = [text[i:i+2000] for i in range(0, len(text), 1800)]
            processed_data = {"source_type": "PDF", "content_chunks": chunks}
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f)
        st.success("One-time pre-processing complete!")
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# --- Main Streamlit UI (Reverted to the simpler layout) ---
st.title("🤖 AI CFO Forecasting Dashboard")
st.markdown("Generate visual financial trends, forecasts, and risk analysis from your documents.")
processed_data = None

st.header("1. Choose Your Data Source", divider='gray')
source_option = st.radio("Select an option:", ("Use Instant Sample (Zomato Annual Report)", "Upload a new file"), horizontal=True, label_visibility="collapsed")

if source_option == "Use Instant Sample (Zomato Annual Report)":
    processed_data = get_or_create_sample_chunks(SAMPLE_PDF_PATH, PREPROCESSED_SAMPLE_PATH)
else:
    uploaded_file = st.file_uploader("Upload a PDF, CSV, or Excel file", type=['pdf', 'csv', 'xlsx'])
    if uploaded_file:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        with st.spinner(f"Processing '{uploaded_file.name}'..."):
            if file_extension == '.pdf':
                processed_data = process_uploaded_pdf(uploaded_file)
            else:
                processed_data = process_structured_file(uploaded_file)

if processed_data:
    st.header("2. Generate Dashboard", divider='gray')
    st.success("✅ Data is ready for analysis.")
    
    if st.button("✨ Generate Financial Dashboard", type="primary", use_container_width=True):
        if processed_data.get("source_type") == "Structured File (CSV/Excel)":
            # For CSV, the data is already processed, so build the dashboard directly
            build_dashboard(processed_data)
        else: 
            # For PDF, run the AI extraction first
            system_prompt, user_prompt = create_master_prompt(processed_data)
            with st.spinner("🤖 AI Extractor is analyzing the document..."):
                extracted_data = run_gemini_extraction(system_prompt, user_prompt, GEMINI_API_KEY)
            
            if extracted_data:
                st.success("Data extraction complete!")
                build_dashboard(extracted_data)
                with st.expander("View Raw Extracted JSON Data"):
                    st.json(extracted_data)
            else:
                st.error("Could not extract financial data to build the dashboard.")

