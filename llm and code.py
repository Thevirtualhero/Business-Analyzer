import streamlit as st
import requests
import json
import re
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Text Preprocessing/Cleaning Function (Retained) ---
def text_clean_special_chars(text):
    """
    Removes special characters, punctuation, and newlines, keeping only letters, 
    numbers, and spaces.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', ' ', text)
    text = re.sub('[^a-zA-Z0-9\s]', '', text)
    text = re.sub('\s+', ' ', text).strip()
    return text

# --- 2. Streamlit App Configuration ---
st.set_page_config(
    page_title="Datastax Business Insight Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3. Custom CSS for Dark Theme & Banking Look (Retained) ---
st.markdown("""
<style>
/* 1. FORCE DARK BACKGROUND */
.main {
    background-color: #1a1a1a; /* Dark background similar to the banking app */
}
.stSidebar {
    background-color: #1a1a1a !important; /* Ensure sidebar is also dark */
}
/* Ensure ALL text is visible on the dark background */
h1, h2, h3, h4, h5, p, div, label, .stMarkdown, .stMetricValue, .stAlert > div > div > div {
    color: #e0e0e0 !important; /* Light gray text for readability */
}

/* 2. CARD STYLING (Light cards on dark background) */
div[data-testid="stVerticalBlock"] > div:not(.stSpinner) {
    background-color: #2b2b2b; /* Slightly lighter card background */
    padding: 1rem;
    border-radius: 0.75rem;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4); /* Deeper shadow for depth */
    margin-bottom: 1rem;
    border: 1px solid #333333; /* Darker border */
}

/* Specific styling for the Main Input area */
div[data-testid="stVerticalBlock"]:has(h3:contains("Customer Review / Data Input")) > div {
    background-color: #2b2b2b; 
}

/* 3. INPUT BOX FIXES (Retained from your code but ensured to look good on dark) */
.stTextArea div[data-baseweb="textarea"] textarea { color: #e0e0e0 !important; }
.stTextArea div[data-baseweb="textarea"] { background-color: #3b3b3b !important; border: 1px solid #555555 !important; }

/* 4. METRIC AND SENTIMENT COLORS (Ensures color contrast) */
.positive-sentiment { color: #10b981 !important; } /* Green */
.negative-sentiment { color: #ef4444 !important; } /* Red */
.neutral-sentiment { color: #f59e0b !important; } /* Orange/Yellow */

/* General button styles */
.stButton>button { width: 100%; border-radius: 0.5rem; height: 3.5rem; font-size: 1.2rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# --- 5. API Configuration ---
API_URL = "https://api.langflow.astra.datastax.com/lf/f507cbcc-8675-4098-a435-23fd15456e5f/api/v1/run/270d96f3-5757-4f34-a4e2-3ac7e457794b"
AUTH_TOKEN = "AstraCS:nfklwJMgMtbRfxZXpOUuHxOM:29012f28a36978c8c4c33c9ab1f213f3cc13d8482354957540ad25ec6784c016" 

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {AUTH_TOKEN}"
}

# --- 6. Visualization Helper Functions (Retained) ---
def generate_sentiment_chart(current_sentiment):
    sentiment_lower = current_sentiment.lower()
    if 'positive' in sentiment_lower: data_label, color = 'Positive', '#10b981'
    elif 'negative' in sentiment_lower: data_label, color = 'Negative', '#ef4444'
    else: data_label, color = 'Neutral', '#f59e0b'
    
    sentiment_counts = pd.Series([100], index=[data_label])
    plt.rcParams['text.color'] = '#e0e0e0' 
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        sentiment_counts.values,
        labels=[f'{data_label} 100%'], 
        startangle=90,
        colors=[color], 
        textprops={'fontsize': 14, 'fontweight': 'bold', 'color': '#e0e0e0'}
    )
    fig.patch.set_alpha(0.0) 
    ax.axis('equal'); ax.set_title('Current Review Sentiment', color='#e0e0e0', fontsize=14)
    st.pyplot(fig)

def generate_theme_chart(theme_string):
    if not theme_string or theme_string.lower() == 'none': st.info("No recurring themes found in the current review."); return
    topics = [t.strip().capitalize() for t in theme_string.split(',') if t.strip()]
    if not topics: return
    theme_counts = pd.Series([1] * len(topics), index=topics)
    plt.rcParams['text.color'] = '#e0e0e0'
    fig, ax = plt.subplots(figsize=(7, 4))
    theme_counts.plot(kind='barh', ax=ax, color='#3b82f6')
    ax.set_title('Themes Found in Current Review', color='#e0e0e0', fontsize=14)
    ax.set_xlabel('Presence (1 = Yes)', color='#e0e0e0')
    ax.tick_params(axis='x', colors='#e0e0e0'); ax.tick_params(axis='y', colors='#e0e0e0'); ax.invert_yaxis()
    fig.patch.set_alpha(0.0)
    ax.set_facecolor('#2b2b2b')
    st.pyplot(fig)


# --- 7. Function to Extract Structured Output (FIXED PARSER - Retained) ---
def extract_results(response_text):
    try:
        response_json = json.loads(response_text)
        raw_output = None
        
        try: raw_output = response_json['outputs'][0]['outputs'][0]['outputs']['message']['message']
        except (KeyError, IndexError, TypeError):
            try: raw_output = response_json['outputs'][0]['outputs'][0]['results']['message']['text']
            except (KeyError, IndexError, TypeError):
                try: raw_output = response_json['outputs'][0]['messages'][0]['message']
                except (KeyError, IndexError, TypeError): pass

        if not raw_output:
            raise ValueError("Raw output text is empty. Check API URL or response path.")
        
        regex = r"1\. STICKER SENTIMENT:\s*(.*?)\n\n2\. EXPERT REVIEW TOPICS:\s*(.*?)\n\n3\. SUMMARIZED INSIGHTS:\s*(.*?)\n\n4\. OPERATIONAL IMPROVEMENT SUGGESTION:\s*(.*?)\n\n5\. TREND-BASED BUSINESS RECOMMENDATION:\s*(.*?)\n\n6\. COMPETITOR COMPARISON HIGHLIGHT:\s*(.*?)(\Z)"
        
        match = re.search(regex, raw_output, re.DOTALL)
        
        if match and match.group(1):
            return {
                "Sentiment": match.group(1).strip(),
                "Topics": match.group(2).strip(),
                "Summary": match.group(3).strip(),
                "Operational_Rec": match.group(4).strip(),
                "Trend_Rec": match.group(5).strip(),
                "Competitor_Rec": match.group(6).strip(),
            }
        
        return {"Error": "Parsing failed. LLM did not follow the exact 1.-6. structure."}

    except Exception as e:
        st.error(f"An unexpected error occurred during parsing: {e}")
        return {"Error": f"Exception: {e}"}


# --- 8. Streamlit UI and Logic ---

st.title("Datastax Business Insight Dashboard 🚀")
st.markdown("Analyze customer feedback to generate 6-point business insights and strategic recommendations.")

# --- Input Container (Simplified Main Column) ---
st.subheader("📝 Customer Review / Data Input")

# Use a single column now that the sidebar panel is removed
with st.container():
    review_input = st.text_area(
        "Enter a customer review (e.g., feedback, sales data, competitor mentions):", 
        value="The new widget we bought failed after only two days. This is poor product quality. Our records show the return rate for this specific widget model has tripled (3x) over the last four weeks.",
        height=150,
        key="review_input_area"
    )
    
    # NLP Cleaning step
    cleaned_review_input = text_clean_special_chars(review_input)
    
    # Button in its own column for centered appearance (retained from original structure)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2: 
        if st.button("✨ Generate 6-Point Analysis", type="primary"):
            
            # --- API Call Logic ---
            payload = {"input_value": cleaned_review_input, "output_type": "chat", "input_type": "chat"}
            
            results_placeholder = st.empty() 
            with st.spinner("Processing with Datastax Agent..."):
                try:
                    response = requests.post(API_URL, json=payload, headers=HEADERS)
                    response.raise_for_status() 
                    
                    results = extract_results(response.text)
                    
                    if "Error" not in results:
                        st.session_state['results'] = results
                    else:
                        st.error(f"Error: {results['Error']}. Check console for raw output.")
                        
                except requests.exceptions.RequestException as e:
                    st.error(f"API Connection Error: Could not connect to Datastax Flow. Details: {e}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")


# --- Results Display (Conditional) ---
if 'results' in st.session_state:
    results = st.session_state['results']
    sentiment_value = results["Sentiment"].upper()
    sentiment_color_class = "neutral-sentiment"
    if "POSITIVE" in sentiment_value: sentiment_color_class = "positive-sentiment"
    elif "NEGATIVE" in sentiment_value: sentiment_color_class = "negative-sentiment"

    # --- NEW DOWNLOAD BUTTON FUNCTIONALITY ---
    
    # Prepare data for download: convert the single result dictionary into a DataFrame for CSV export
    results_df = pd.DataFrame([results])
    
    @st.cache_data
    def convert_df_to_csv(df):
        # Function to convert DataFrame to CSV string
        return df.to_csv(index=False).encode('utf-8')

    csv_data = convert_df_to_csv(results_df)
    
    # Display the download button
    st.markdown("---")
    st.markdown("### 💾 Download Analysis")
    st.download_button(
        label="⬇️ Download 6-Point Analysis (CSV)",
        data=csv_data,
        file_name='single_review_analysis.csv',
        mime='text/csv',
    )
    # --- END NEW DOWNLOAD BUTTON ---
    
    
    st.markdown("---")
    st.markdown("### 📊 Business Overview")
    
    # 1. Summary Metrics Section 
    col_a1, col_a2, col_a3 = st.columns(3)
    
    with col_a1:
        with st.container():
            st.markdown("##### Current Sentiment")
            st.markdown(f"<p class='stMetricValue {sentiment_color_class}' style='font-size:2rem; font-weight:bold;'>{sentiment_value}</p>", unsafe_allow_html=True)

    with col_a2:
        with st.container():
            st.markdown("##### Extracted Topics")
            st.info(results["Topics"])

    with col_a3:
        with st.container():
            st.markdown("##### Summarized Insights")
            st.info(results["Summary"])


    # 2. Visualizations and Analysis
    st.markdown("---")
    st.markdown("### 📈 Deep Dive Analysis")

    viz_col1, viz_col2 = st.columns([1, 1])
    
    with viz_col1:
        with st.container():
            st.subheader("Current Sentiment Chart")
            generate_sentiment_chart(results["Sentiment"]) 
    
    with viz_col2:
        with st.container():
            st.subheader("Themes Found in Review")
            generate_theme_chart(results["Topics"])


    # 3. Recommendation Section
    st.markdown("---")
    st.subheader("💡 Actionable Recommendations")

    col_r1, col_r2, col_r3 = st.columns(3)
    
    with col_r1:
        with st.container():
            st.markdown("##### 4. Operational Improvement")
            if "not applicable" in results["Operational_Rec"].lower():
                st.success(results["Operational_Rec"])
            else:
                st.warning(results["Operational_Rec"])
    
    with col_r2:
        with st.container():
            st.markdown("##### 5. Trend-based Recommendation")
            st.info(results["Trend_Rec"])
        
    with col_r3:
        with st.container():
            st.markdown("##### 6. Competitor Comparison")
            st.info(results["Competitor_Rec"])