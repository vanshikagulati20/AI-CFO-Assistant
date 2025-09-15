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
    """
    Processes a newly uploaded finance-related CSV or Excel file with auto-detection
    and correct yearly aggregation.
    """
    # The 'try' block starts here and must have a corresponding 'except' at the end.
    try:
        st.write("Loading structured data...")
        uploaded_file.seek(0)

        # Load file
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file, encoding="utf-8", on_bad_lines="skip")
        elif uploaded_file.name.lower().endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload a CSV or Excel file.")
            return None

        # --- 1. Normalize Column Names ---
        df.columns = df.columns.str.strip().str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
        st.write("Cleaned columns detected:", df.columns.tolist())

        # --- 2. Define Synonyms and Find Columns ---
        synonyms = {
            "date": ["date", "transactiondate", "day", "month", "year", "period"],
            "revenue": ["revenue", "income", "sales", "earning", "turnover", "grosssales", "netrevenue"],
            "profit": ["profit", "netprofit", "gain", "netincome", "earningafterexpenses"],
            "expense": ["expense", "cost", "spend", "expenditure", "loss"],
        }

        def find_col(candidates):
            for c in candidates:
                if c in df.columns:
                    return c
            return None

        date_col = find_col(synonyms["date"])
        revenue_col = find_col(synonyms["revenue"])
        profit_col = find_col(synonyms["profit"])
        expense_col = find_col(synonyms["expense"])
        
        if date_col: st.info(f"✅ Mapped date column: '{date_col}'")
        if revenue_col: st.info(f"✅ Mapped revenue column: '{revenue_col}'")
        if profit_col: st.info(f"✅ Mapped profit column: '{profit_col}'")
        if expense_col: st.info(f"✅ Mapped expense column: '{expense_col}'")

        # --- 3. Validate if at least one key financial column is found ---
        if not any([revenue_col, profit_col, expense_col]):
             st.error("No key financial columns (revenue, profit, expense) found. Please check your file.")
             return None

        # --- 4. Convert Financial Columns to Numeric ---
        if revenue_col:
            df[revenue_col] = pd.to_numeric(df[revenue_col], errors='coerce').fillna(0)
        if profit_col:
            df[profit_col] = pd.to_numeric(df[profit_col], errors='coerce').fillna(0)
        if expense_col:
            df[expense_col] = pd.to_numeric(df[expense_col], errors='coerce').fillna(0)

        # --- 5. Calculate Profit if not directly available ---
        if profit_col:
            df['calculated_profit'] = df[profit_col]
        elif revenue_col and expense_col:
            st.info("Profit column not found. Calculating profit as Revenue - Expense.")
            df['calculated_profit'] = df[revenue_col] - df[expense_col]
        elif revenue_col:
            st.warning("Profit and Expense columns not found. Using Revenue as a fallback for Profit.")
            df['calculated_profit'] = df[revenue_col]
        else:
            df['calculated_profit'] = 0

        # --- 6. Aggregate Data by Year ---
        historical_performance = []
        if date_col:
            df['year'] = pd.to_datetime(df[date_col], errors='coerce').dt.year
            df.dropna(subset=['year'], inplace=True)
            if not df.empty:
                df['year'] = df['year'].astype(int)
                yearly_data = df.groupby('year').agg(
                    revenue_crore=(revenue_col, 'sum') if revenue_col else pd.NamedAgg(column='year', aggfunc=lambda x: 0),
                    profit_crore=('calculated_profit', 'sum')
                ).reset_index()
                historical_performance = yearly_data.to_dict('records')
        else:
            st.warning("No date column found. Aggregating all data into a single record.")
            total_revenue = df[revenue_col].sum() if revenue_col else 0
            total_profit = df['calculated_profit'].sum()
            historical_performance.append({
                "year": "N/A", "revenue_crore": total_revenue, "profit_crore": total_profit,
            })
            
        # --- 7. Debugging Section ---
        if not historical_performance:
            st.error("Could not extract any valid historical performance data after processing.")
            st.subheader("🕵️‍♂️ Debugging Info")
            st.warning("The final `historical_performance` list is empty. Here's the raw data pandas processed:")
            st.dataframe(df)
            return None
        
        st.subheader("🕵️‍♂️ Debugging Info (Success Case)")
        st.write("Final calculated `historical_performance` object:")
        st.write(historical_performance)

        # ---- Construct final result ----
        return {
            "historical_performance": historical_performance,
            "revenue_breakdown_crore": [],
            "future_risks": ["Manual analysis needed for raw transaction data."],
            "future_opportunities": ["Scaling customer base and improving margins are key opportunities."]
        }

    # This 'except' block was missing or misplaced, causing the SyntaxError.
    except Exception as e:
        st.error(f"A critical error occurred in process_structured_file: {e}")
        return None

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

# --- Main Streamlit UI (CORRECTED with State Reset) ---
st.title("🤖 AI CFO Forecasting Dashboard")
st.markdown("Generate visual financial trends, forecasts, and risk analysis from your documents.")

# Initialize session_state if it doesn't exist
if 'dashboard_data' not in st.session_state:
    st.session_state.dashboard_data = None

st.header("1. Choose Your Data Source", divider='gray')
source_option = st.radio(
    "Select an option:",
    ("Use Instant Sample (Zomato Annual Report)", "Upload a new file"),
    horizontal=True,
    label_visibility="collapsed"
)

processed_data_for_ai = None
processed_data_direct = None

if source_option == "Use Instant Sample (Zomato Annual Report)":
    # Reset state when switching to the sample
    st.session_state.dashboard_data = None
    processed_data_for_ai = get_or_create_sample_chunks(SAMPLE_PDF_PATH, PREPROCESSED_SAMPLE_PATH)
else:
    uploaded_file = st.file_uploader("Upload a PDF, CSV, or Excel file", type=['pdf', 'csv', 'xlsx', 'xls'])
    
    # THE FIX: Reset the dashboard state as soon as a new file is uploaded
    if uploaded_file is not None:
        st.session_state.dashboard_data = None  # <-- THIS IS THE KEY FIX
        
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        with st.spinner(f"Processing '{uploaded_file.name}'..."):
            if file_extension == '.pdf':
                processed_data_for_ai = process_uploaded_pdf(uploaded_file)
            else:
                processed_data_direct = process_structured_file(uploaded_file)

# Logic to generate the dashboard
if processed_data_for_ai or processed_data_direct:
    st.header("2. Generate Dashboard", divider='gray')
    st.success("✅ Data is ready for analysis.")

    if st.button("✨ Generate Financial Dashboard", type="primary", use_container_width=True):
        if processed_data_direct:
            st.session_state.dashboard_data = processed_data_direct
        elif processed_data_for_ai:
            system_prompt, user_prompt = create_master_prompt(processed_data_for_ai)
            with st.spinner("🤖 AI Extractor is analyzing the document..."):
                extracted_data = run_gemini_extraction(system_prompt, user_prompt, GEMINI_API_KEY)
                if extracted_data:
                    st.success("Data extraction complete!")
                    st.session_state.dashboard_data = extracted_data
                else:
                    st.error("Could not extract financial data to build the dashboard.")
                    st.session_state.dashboard_data = None

# Logic to display the dashboard (now safely separated)
if st.session_state.dashboard_data:
    build_dashboard(st.session_state.dashboard_data)
    with st.expander("View Raw Extracted JSON Data"):
        st.json(st.session_state.dashboard_data)
