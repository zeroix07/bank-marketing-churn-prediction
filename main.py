import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os 
from datetime import timedelta, datetime
from io import StringIO
import json
import requests
import time
from dotenv import load_dotenv # Import load_dotenv

# Load environment variables from .env file immediately
load_dotenv()

# --- 0. CONFIGURATION & SETUP ---
MODEL_FILE = 'random_forest_frozen.pkl'
LATEST_DATE_FOR_RECENCY = pd.to_datetime('2024-12-31')
# Load API Key from environment variable (will be None if not set)
API_KEY = os.environ.get("GEMINI_API_KEY") 

# --- 1. MODEL & DATA PROCESSING FUNCTIONS (MUST MATCH TRAINING) ---

@st.cache_resource
def load_model():
    """Loads the trained Random Forest pipeline."""
    try:
        pipeline = joblib.load(MODEL_FILE)
        return pipeline
    except FileNotFoundError:
        st.error(f"Error: Model file '{MODEL_FILE}' not found. Please ensure it is in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def aggregate_new_transactions(df_transactions):
    """Aggregates transactional data to a customer level (RFM + demographics)."""
    
    # Ensure Transaction_Date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_transactions['Transaction_Date']):
        df_transactions['Transaction_Date'] = pd.to_datetime(df_transactions['Transaction_Date'])

    agg_dict = {
        'Transaction_Date': [('Recency', lambda x: (LATEST_DATE_FOR_RECENCY - x.max()).days)],
        'Transaction_Amount': [('Monetary', 'sum'), ('Frequency', 'count')], 
        'age': 'first', 'balance': 'first', 'duration': 'first', 'campaign': 'first',
        'pdays': 'first', 'previous': 'first', 
        'job': 'first', 'marital': 'first', 'education': 'first', 
        'default': 'first', 'housing': 'first', 'loan': 'first', 
        'contact': 'first', 'month': 'first', 'poutcome': 'first',
        'y': 'first' # Include a dummy numerical column 'y_numeric' to match training features
    }

    df_customer_info = df_transactions.groupby('Customer_ID').agg(agg_dict).reset_index()

    # Flatten columns (MUST match training features exactly)
    df_customer_info.columns = ['Customer_ID', 'Recency', 'Monetary', 'Frequency', 'age', 
                               'balance', 'duration', 'campaign', 'pdays', 'previous', 'y',
                               'job', 'marital', 'education', 'default', 'housing', 'loan', 
                               'contact', 'month', 'poutcome']
    
    # Apply the crucial numerical fix (-1 to 0 in pdays)
    df_customer_info['pdays'] = df_customer_info['pdays'].replace(-1, 0)
    
    # Add dummy 'y_numeric' column to align with training features (FIX from previous ValueError)
    df_customer_info['y_numeric'] = 0 

    return df_customer_info

def predict_churn(df_customer_info, pipeline):
    """Prepares features and runs the prediction."""
    
    features_for_model = ['Recency', 'Monetary', 'Frequency', 'age', 'balance', 
                          'duration', 'campaign', 'pdays', 'previous', 
                          'job', 'marital', 'education', 'default', 'housing', 
                          'loan', 'contact', 'month', 'poutcome', 
                          'y_numeric'] # Includes the dummy numerical column

    X_new = df_customer_info[features_for_model]
    
    predictions = pipeline.predict(X_new)
    
    df_customer_info['Predicted_Label'] = predictions
    df_customer_info['Predicted_Segment'] = df_customer_info['Predicted_Label'].map(
        {1: 'Mass Market/Resistor (High Churn)', 0: 'Champion/Potentialist (Low Churn)'}
    )
    
    # Map the Predicted_Label back to the specific DBSCAN Cluster Names (e.g., Champions, Resistors)
    # This requires more advanced logic, but for simplified display, we can infer the action group:
    df_customer_info['Action_Group'] = df_customer_info['Predicted_Segment'].apply(
        lambda x: 'High Churn Risk' if 'High Churn' in x else 'Low Churn Risk/High Value'
    )

    df_results = df_customer_info[['Customer_ID', 'Recency', 'Frequency', 'Monetary', 'Predicted_Segment', 'Action_Group']]
    return df_results


# --- 2. GEMINI CHATBOT LOGIC ---

SEGMENT_PROFILES = """
### Customer Segment Profiles:
1. **👑 Champions (Low Churn/High Value):** Highest-Value, Highly Responsive. Recommended Action: Reward and Retain.
2. **💼 Seasoned Clientele (Low Churn/High Value):** Older & Responsive. Recommended Action: Tailor Messaging (Retirement/Stability products).
3. **🌱 Potentialists (Low Churn/High Value):** Emerging Value. Recommended Action: Test & Convert (Stronger incentives).
4. **🌐 Mass Market (High Churn Risk):** The Baseline. Recommended Action: Efficiency (Automated, low-cost campaigns).
5. **🚧 Resistors (High Churn Risk):** Non-Responsive/At-Risk. Recommended Action: Exclusion List (Do Not Contact).

Note: The model predicts the **Risk Group** (High Churn = Mass Market/Resistor, Low Churn = Champion/Seasoned/Potentialist).
"""

def call_gemini_api(prompt, data_context=""):
    """Calls the Gemini API to generate content."""
    
    if not API_KEY:
        return "⚠️ Gemini API key is required. Please set the GEMINI_API_KEY environment variable."

    # Construct the system prompt including detailed segment profiles
    system_prompt = (
        "You are an expert Marketing Analyst specializing in RFM segmentation and predictive modeling. "
        "Your responses must be highly actionable and professional. "
        f"{SEGMENT_PROFILES}\n\n"
        "Your tasks are to: 1) Summarize the predicted customer data based on risk groups; 2) Perform EDA/Correlation analysis if requested; 3) Generate specific marketing strategies based on the segment profiles above. "
        "Always provide Markdown-formatted output, including tables and bolding."
    )

    # Combine context and user query
    full_query = (
        f"PREDICTED CUSTOMER DATA:\n---\n{data_context}\n---\n"
        f"USER REQUEST: {prompt}"
    )

    # API call setup
    apiUrl = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"
    
    payload = {
        "contents": [{"parts": [{"text": full_query}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }
    
    headers = {'Content-Type': 'application/json'}
    
    # Implement exponential backoff for robustness
    max_retries = 3
    delay = 1
    
    for attempt in range(max_retries):
        try:
            response = requests.post(f"{apiUrl}?key={API_KEY}", headers=headers, json=payload)
            response.raise_for_status() 
            
            result = response.json()
            text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'No response text found.')
            return text

        except requests.exceptions.HTTPError as e:
            if response.status_code == 429 and attempt < max_retries - 1:
                import time
                time.sleep(delay)
                delay *= 2
            else:
                return f"⚠️ API Error ({response.status_code}): {e}. Check API key and quota."
        except Exception as e:
            return f"⚠️ An unexpected error occurred during API call: {e}"
    return "API call failed after multiple retries due to rate limiting."

# --- 3. STREAMLIT UI COMPONENTS ---

def handle_prediction(df_new_transactions, pipeline):
    """Handles the prediction and session state update."""
    with st.spinner("Aggregating RFM and predicting segments..."):
        try:
            df_customer_info = aggregate_new_transactions(df_new_transactions)
            df_results = predict_churn(df_customer_info, pipeline)
            
            st.subheader("2. Prediction Results")
            st.dataframe(df_results, use_container_width=True)
            
            # Store results for Chatbot analysis
            st.session_state['prediction_results'] = df_results
            st.session_state['prediction_analysis_context'] = df_customer_info.to_markdown(index=False)
            
            st.success("Prediction complete! Use the AI Chatbot tab for analysis and strategy.")

        except Exception as e:
            st.error(f"Error during prediction or data processing: {e}. Check your column names and data types.")
            st.exception(e)

def display_prediction_tab(pipeline):
    st.header("Customer Segmentation & Churn Prediction")
    st.markdown("Predict the churn propensity (High Risk vs. Low Risk) of new customers.")

    if not pipeline:
        return

    st.subheader("1. Input New Transactional Data")
    
    # --- Updated Radio Button Options ---
    input_method = st.radio(
        "Choose Data Input Method:",
        ("Use Sample Data", "Enter Data via Form") # Removed "Upload Custom CSV File"
    )
    
    df_new_transactions = None
    
    if input_method == "Use Sample Data":
        st.info("Using embedded sample data (3 customers) for a quick demonstration.")
        sample_data_str = """Customer_ID,Transaction_Date,Transaction_Amount,age,job,marital,education,balance,housing,loan,contact,day,month,duration,campaign,pdays,previous,poutcome,y,default,Subscription_Status
100004,2024-12-28,500.00,35,management,married,tertiary,3000,yes,no,cellular,15,jan,300,1,-1,0,unknown,no,no,no
100004,2024-12-10,100.00,35,management,married,tertiary,3000,yes,no,cellular,15,jan,300,1,-1,0,unknown,no,no,no
100005,2024-05-01,50.00,50,blue-collar,married,secondary,50,yes,yes,telephone,10,may,150,5,180,2,failure,no,no,no
100005,2024-03-01,20.00,50,blue-collar,married,secondary,50,yes,yes,telephone,10,may,150,5,180,2,failure,no,no,no
100006,2024-10-01,1500.00,28,student,single,tertiary,10,no,no,cellular,22,oct,90,1,-1,0,unknown,yes,no,yes"""
        
        try:
            df_new_transactions = pd.read_csv(StringIO(sample_data_str))
        except Exception:
            pass 

        st.text_area("Sample Data Preview:", sample_data_str, height=150, disabled=True)
        
        if st.button("Predict Churn Segments (Sample)"):
            if df_new_transactions is not None:
                handle_prediction(df_new_transactions, pipeline)
            else:
                st.warning("Could not read sample data.")


    elif input_method == "Enter Data via Form":
        st.subheader("Simulate Single Customer Transactions")
        
        with st.form("single_customer_form"):
            st.markdown("**1. Customer Demographics & Financials**")
            col1, col2, col3 = st.columns(3)
            with col1:
                form_age = st.number_input("Age", min_value=18, max_value=100, value=35)
                form_job = st.selectbox("Job Type", options=["management", "blue-collar", "technician", "admin.", "services", "retired", "self-employed", "unemployed", "entrepreneur", "housemaid", "student", "unknown"], index=0)
                form_marital = st.selectbox("Marital Status", options=["married", "single", "divorced"], index=0)
                form_education = st.selectbox("Education Level", options=["tertiary", "secondary", "primary", "unknown"], index=0)
            with col2:
                form_balance = st.number_input("Average Balance (€)", min_value=-5000, max_value=100000, value=1500)
                form_housing = st.selectbox("Housing Loan", options=["yes", "no"], index=1)
                form_loan = st.selectbox("Personal Loan", options=["yes", "no"], index=1)
                form_default = st.selectbox("Credit Default", options=["no", "yes"], index=0)
            with col3:
                # Force last transaction date to be in the past
                form_contact_date = st.date_input("Last Transaction Date", value=datetime.now().date() - timedelta(days=30))
                form_contact_type = st.selectbox("Contact Type", options=["cellular", "telephone", "unknown"], index=0)
                form_duration = st.number_input("Last Contact Duration (s)", min_value=0, max_value=3000, value=180)

            st.markdown("---")
            st.markdown("**2. Campaign & Transaction History (Simulated)**")
            col4, col5, col6 = st.columns(3)
            with col4:
                form_num_transactions = st.number_input("Number of Transactions (F)", min_value=1, max_value=10, value=3, help="Simulated Frequency (F)")
                form_total_monetary = st.number_input("Total Monetary Value (€)", min_value=10.0, value=300.0, help="Simulated Monetary Value (M)")
            with col5:
                form_campaign = st.number_input("Campaign Contacts", min_value=1, max_value=20, value=2, help="Contacts during current campaign")
                form_previous = st.number_input("Previous Contacts", min_value=0, max_value=10, value=1, help="Contacts before current campaign")
            with col6:
                form_pdays = st.number_input("Days Since Previous Contact (pdays)", min_value=-1, max_value=365, value=90, help="-1 if no previous contact")
                form_poutcome = st.selectbox("Previous Outcome", options=["unknown", "failure", "success", "other"], index=0)
                
            submitted = st.form_submit_button("Generate & Predict")
            
            if submitted:
                # --- Create Synthetic Transactional Data ---
                try:
                    # Calculate required synthetic values
                    avg_monetary = form_total_monetary / form_num_transactions
                    
                    simulated_data = []
                    for i in range(int(form_num_transactions)):
                        # Simulate different historical transaction dates and amounts
                        simulated_data.append({
                            'Customer_ID': 999999,
                            'Transaction_Date': form_contact_date - timedelta(days=i * 10),
                            'Transaction_Amount': avg_monetary * np.random.uniform(0.9, 1.1),
                            # Copy the static form fields for each row
                            'age': form_age, 'job': form_job, 'marital': form_marital, 
                            'education': form_education, 'balance': form_balance, 'housing': form_housing,
                            'loan': form_loan, 'contact': form_contact_type, 'day': form_contact_date.day, 
                            'month': form_contact_date.strftime('%b').lower(), 'duration': form_duration, 
                            'campaign': form_campaign, 'pdays': form_pdays, 'previous': form_previous, 
                            'poutcome': form_poutcome, 'y': 'no', 'default': form_default, 
                            'Subscription_Status': 'no'
                        })
                    
                    df_new_transactions = pd.DataFrame(simulated_data)
                    st.success(f"Generated {form_num_transactions} synthetic transactions for prediction.")
                    
                    # Run prediction immediately after data generation
                    handle_prediction(df_new_transactions, pipeline)

                except ZeroDivisionError:
                    st.error("Error: Number of Transactions (F) must be greater than zero.")
                except Exception as e:
                    st.error(f"Error during form data processing: {e}")
                    st.exception(e)


def display_chatbot_tab():
    st.header("AI Marketing Analyst Chatbot")
    st.markdown("Ask the Gemini model to perform EDA, generate marketing strategies, or summarize the predicted customer data.")

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "Hello! I am your AI Marketing Analyst. I have been loaded with the segment definitions and your latest prediction data. Ask me for a **strategy** for the 'High Churn Risk' group, or ask to analyze the data."})

    # Check if prediction results are available
    data_context = st.session_state.get('prediction_analysis_context', "No specific customer data has been predicted yet. I can only answer general marketing questions.")
    
    if 'prediction_results' in st.session_state:
        results_markdown = st.session_state['prediction_results'].to_markdown(index=False)
        data_context = f"PREDICTED CHURN RESULTS (Risk Group and RFM Metrics):\n---\n{results_markdown}\n---\n" + data_context
        st.info("Prediction data is loaded into the chat context.")
    else:
        st.warning("Please run a prediction in the 'Prediction Interface' tab first to load customer data for the AI analysis.")


    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask for analysis, correlation, or strategy..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing data and generating response..."):
                response = call_gemini_api(prompt, data_context)
                st.markdown(response)
                
            st.session_state.messages.append({"role": "assistant", "content": response})

# --- 4. MAIN APP FUNCTION ---

def main():
    st.set_page_config(layout="wide", page_title="RFM Churn Prediction & Gemini Analyst")
    st.title("RFM-DBSCAN Churn Prediction Dashboard")

    # Load the model once
    pipeline = load_model()

    if pipeline:
        # Create Tabs
        tab1, tab2 = st.tabs(["📊 Prediction Interface", "🤖 AI Marketing Analyst Chatbot"])

        with tab1:
            display_prediction_tab(pipeline)

        with tab2:
            display_chatbot_tab()

if __name__ == "__main__":
    # Ensure the required synchronous fetch functions (requests) are available for simulation
    try:
        import requests
        
        # Check for dotenv installation
        try:
            from dotenv import load_dotenv
        except ImportError:
            st.error("The 'python-dotenv' library is required to load your .env file. Please install it: `pip install python-dotenv`")
        else:
            # Run main if requests and dotenv are available
            main()
            
    except ImportError:
        st.error("The 'requests' library is required for the simulated API calls. Please install it: `pip install requests`")