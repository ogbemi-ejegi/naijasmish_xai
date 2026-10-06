import pandas as pd
import os
import time
import re
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import sys

# Add parent directory to path so we can import core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.crew_logic import run_inspection
from utils.preprocess import preprocess_sms
from database.chroma_engine import init_db

load_dotenv()

def extract_verdict(final_output):
    """Robustly extract the final verdict"""
    final_output = str(final_output).lower()
    
    # Try to find probability explicitly
    prob_match = re.search(r'(?:threat_probability|probability)[\"\'\s:]+(\d+)', final_output)
    probability = int(prob_match.group(1)) if prob_match else 0
    
    # Try to find true/false explicitly
    verdict_match = re.search(r'(?:final_verdict|verdict)[\"\'\s:]+(true|false)', final_output)
    
    is_phishing = False
    if verdict_match:
        is_phishing = verdict_match.group(1) == 'true'
    elif probability >= 70:
        is_phishing = True
    elif "true" in final_output and "false" not in final_output:
        is_phishing = True
        
    return 1 if is_phishing else 0

def main():
    print("Initializing local ChromaDB with V2 dataset...")
    init_db()
    
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'validation_100_set.csv')
    if not os.path.exists(csv_path):
        print("validation_100_set.csv not found. Run baseline_eval.py first.")
        return
        
    df = pd.read_csv(csv_path)
    
    y_true = df['Spam'].tolist()
    y_pred = []
    
    print("Starting Multi-Agent Evaluation on 100 messages...")
    print("NOTE: This will take roughly 20-30 minutes to avoid Google API rate limits (HTTP 429).")
    
    success_count = 0
    
    for idx, row in df.iterrows():
        msg = row['Modified_Message']
        preprocessed = preprocess_sms(msg)
        
        max_retries = 3
        verdict = 0
        
        for attempt in range(max_retries):
            try:
                print(f"Inspecting message {idx+1}/100...")
                results = run_inspection(preprocessed)
                final_output = results["final_result"]
                verdict = extract_verdict(final_output)
                success_count += 1
                
                # Sleep to prevent rate limit (Gemini free tier allows 15 RPM)
                # 3 agents = 3 requests per message. 1 request every 4 seconds = 1 message every 12 seconds
                time.sleep(1)
                break
            except Exception as e:
                print(f"Error on message {idx+1}, attempt {attempt+1}: {str(e)}")
                time.sleep(20) # Exponential-ish backoff
        
        y_pred.append(verdict)
        
        # Save incremental results just in case it crashes
        df.loc[idx, 'Agent_Prediction'] = verdict
        df.to_csv(csv_path, index=False)
        
    print("\nEvaluation Complete.")
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n=========================================")
    print("      MULTI-AGENT CREW RESULTS           ")
    print("=========================================")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("=========================================\n")

if __name__ == "__main__":
    main()
