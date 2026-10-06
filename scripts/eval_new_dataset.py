import pandas as pd
import os
import time
import re
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.crew_logic import run_inspection
from utils.preprocess import preprocess_sms
from database.chroma_engine import init_db

load_dotenv()

def extract_verdict(final_output):
    final_output = str(final_output).lower()
    prob_match = re.search(r'(?:threat_probability|probability)[\"\'\s:]+(\d+)', final_output)
    probability = int(prob_match.group(1)) if prob_match else 0
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
    print("Initializing local ChromaDB for New Dataset Evaluation...")
    init_db()
    
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'new_dataset.csv')
    df = pd.read_csv(csv_path)
    
    y_true = df['Spam'].tolist()
    y_pred = []
    
    print("Starting REAL Multi-Agent Evaluation on NEW 50 messages...")
    
    for idx, row in df.iterrows():
        msg = row['Message']
        preprocessed = preprocess_sms(msg)
        
        verdict = 0
        try:
            print(f"Inspecting message {idx+1}/50...", flush=True)
            results = run_inspection(preprocessed)
            final_output = results["final_result"]
            verdict = extract_verdict(final_output)
        except Exception as e:
            print(f"Error on message {idx+1}: {str(e)}")
        
        y_pred.append(verdict)
        time.sleep(1) # Safe 1-second delay
        
    print("\nEvaluation Complete!")
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n=========================================")
    print("   MULTI-AGENT GEMINI 2.5 FLASH RESULTS  ")
    print("          (NEW DATASET)                  ")
    print("=========================================")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("Confusion Matrix:\n", cm)
    print("=========================================\n")

if __name__ == "__main__":
    main()
