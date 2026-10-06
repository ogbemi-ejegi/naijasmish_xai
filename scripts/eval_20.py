import pandas as pd
import os
import time
import re
import json
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import sys

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
    print("Initializing local ChromaDB with smaller dataset...")
    init_db()
    
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'validation_100_set.csv')
    df = pd.read_csv(csv_path)
    
    # Select exactly 10 Spam and 10 Ham
    spam_df = df[df['Spam'] == 1].head(10)
    ham_df = df[df['Spam'] == 0].head(10)
    test_df = pd.concat([spam_df, ham_df]).reset_index(drop=True)
    
    y_true = test_df['Spam'].tolist()
    y_pred = []
    
    print("Starting Multi-Agent Evaluation on 20 messages...")
    
    for idx, row in test_df.iterrows():
        msg = row['Modified_Message']
        preprocessed = preprocess_sms(msg)
        
        max_retries = 3
        verdict = 0
        
        for attempt in range(max_retries):
            try:
                print(f"Inspecting message {idx+1}/20 (Ground Truth: {row['Spam']})...")
                results = run_inspection(preprocessed)
                final_output = results["final_result"]
                verdict = extract_verdict(final_output)
                
                print(f" -> Predicted: {verdict}")
                
                # Sleep to prevent rate limit
                time.sleep(12)
                break
            except Exception as e:
                print(f"Error on message {idx+1}, attempt {attempt+1}: {str(e)}")
                time.sleep(15) 
        
        y_pred.append(verdict)
        
    print("\nEvaluation Complete.")
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n=========================================")
    print("      MULTI-AGENT CREW RESULTS (20 MSGS) ")
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
