import streamlit as st
import json
import re
import os
from dotenv import load_dotenv

# Load env variables (e.g. GEMINI_API_KEY)
load_dotenv()

from utils.preprocess import preprocess_sms
from database.chroma_engine import init_db
from core.crew_logic import run_inspection

# Page Config
st.set_page_config(page_title="NaijaSmish-XAI", layout="centered")

# Custom CSS for aesthetics (Web Application Development Prompt Guidelines)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    .main-header {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
        padding-top: 2rem;
    }
    
    .threat-banner-high {
        background: linear-gradient(90deg, #ef4444 0%, #b91c1c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 800;
        font-size: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 1.5rem;
    }
    
    .threat-banner-low {
        background: linear-gradient(90deg, #10b981 0%, #047857 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 800;
        font-size: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 1.5rem;
    }
    
    .summary-panel {
        background: rgba(255, 255, 255, 0.05);
        border-left: 4px solid #8b5cf6;
        padding: 1.5rem;
        border-radius: 0 12px 12px 0;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
        font-size: 1.1rem;
        line-height: 1.6;
    }
    
    .stTextArea textarea {
        background-color: #1e293b;
        color: #f8fafc;
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">NaijaSmish-XAI</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8; margin-bottom: 2rem; font-size: 1.1rem;'>Explainable Multi-Agent SMS Phishing Detector</p>", unsafe_allow_html=True)

# Initialize DB on load
@st.cache_resource(show_spinner="Initializing Database (First run takes 1-2 minutes to download models)...")
def load_database_v2():
    try:
        init_db()
        return True
    except Exception as e:
        st.error(f"Failed to initialize database: {str(e)}")
        return False

db_ready = load_database_v2()

st.subheader("Message Inspection")
sms_input = st.text_area("Enter or paste suspicious Nigerian SMS text here...", height=150, placeholder="e.g. Dear Customer, your BVN has been deactivated. Please call this number urgently...")

if st.button("Execute Multi-Agent Inspection", type="primary", use_container_width=True):
    if not os.getenv("GEMINI_API_KEY"):
        st.error("GEMINI_API_KEY is missing in the .env file. Please add it before running.")
    elif not sms_input.strip():
        st.warning("Please enter an SMS text to inspect.")
    else:
        with st.spinner("Analyzing message with AI Agents..."):
            preprocessed_text = preprocess_sms(sms_input)
            st.info(f"**Preprocessed Text:** {preprocessed_text}")
            
            try:
                results = run_inspection(preprocessed_text)
                
                final_output = results["final_result"]
                
                # Attempt to extract probability (very naive extraction just for the banner)
                # It looks for "threat_probability" or "Probability:" followed by a number
                prob_match = re.search(r'(?:threat_probability|probability).*?(\d+)', final_output, re.IGNORECASE)
                probability = int(prob_match.group(1)) if prob_match else 0
                
                # Naive verdict boolean match
                verdict_match = re.search(r'(?:final_verdict|verdict).*?(true|false)', final_output, re.IGNORECASE)
                is_phishing = True if probability >= 70 else False
                if verdict_match:
                    is_phishing = verdict_match.group(1).lower() == 'true'
                
                st.markdown("---")
                
                if is_phishing or probability >= 70:
                    st.markdown(f'<div class="threat-banner-high">🚨 CRITICAL THREAT DETECTED - PROBABILITY: {probability}%</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="threat-banner-low">✅ NO CONTEXTUAL PHISHING DETECTED - PROBABILITY: {probability}%</div>', unsafe_allow_html=True)
                
                st.markdown("### Explainable AI Summary")
                st.markdown(f'<div class="summary-panel">{final_output}</div>', unsafe_allow_html=True)
                
                with st.expander("Technical Agent Deliberation Traces (Traceability Logs)"):
                    st.markdown("#### Linguistic Agent Output")
                    st.code(results["linguistic_trace"])
                    st.markdown("#### Pattern Agent Output")
                    st.code(results["pattern_trace"])
                    
            except Exception as e:
                st.error(f"An error occurred during inspection: {str(e)}")
