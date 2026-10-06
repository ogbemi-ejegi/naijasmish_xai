import pandas as pd
import os
import requests
import json
import time
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

load_dotenv()
api_key = os.getenv('OPENROUTER_API_KEY')
headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json',
    'HTTP-Referer': 'http://localhost',
    'X-Title': 'NaijaSmish-XAI'
}

def extract_verdict(final_output):
    final_output = str(final_output).lower()
    if 'verdict: true' in final_output or 'phishing' in final_output or 'spam' in final_output:
        return 1
    return 0

def evaluate_message(msg):
    prompt = f"""
    You are a Cybersecurity Meta-Judge. Analyze the following Nigerian SMS message and determine if it is a phishing/scam attempt. 
    Look for structural code-switching, local urgency markers (e.g., 'sharp sharp', 'abeg'), BVN restriction claims, or spoofed shortcodes.
    
    Message: "{msg}"
    
    Output exactly one line: "Verdict: True" if it is spam/phishing, or "Verdict: False" if it is legitimate.
    """
    
    data = {
        'model': 'google/gemini-3.8-flash',
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0.0
    }
    
    try:
        res = requests.post('https://openrouter.ai/api/v1/chat/completions', headers=headers, json=data, timeout=15)
        if res.status_code == 200:
            content = res.json()['choices'][0]['message']['content']
            return extract_verdict(content)
        else:
            print(f"API Error {res.status_code}: {res.text[:100]}")
            return 0
    except Exception as e:
        print(f"Request failed: {str(e)}")
        return 0

from concurrent.futures import ThreadPoolExecutor, as_completed

def main():
    print("Starting Direct OpenRouter Evaluation (Fast Multi-threaded)...", flush=True)
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'validation_100_set.csv')
    df = pd.read_csv(csv_path)
    
    y_true = df['Spam'].tolist()
    y_pred_dict = {}
    
    def evaluate_row(idx, msg):
        return idx, evaluate_message(msg)
        
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(evaluate_row, idx, row['Modified_Message']): idx for idx, row in df.iterrows()}
        for i, future in enumerate(as_completed(futures)):
            idx, verdict = future.result()
            y_pred_dict[idx] = verdict
            if (i+1) % 10 == 0:
                print(f"Processed {i+1}/100...", flush=True)
                
    y_pred = [y_pred_dict[i] for i in range(len(df))]
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n=========================================")
    print("   DIRECT GEMINI 3.8 FLASH RESULTS       ")
    print("=========================================")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("Confusion Matrix:\n", cm)
    print("=========================================\n")

if __name__ == "__main__":
    main()
