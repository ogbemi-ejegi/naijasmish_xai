import pandas as pd
import numpy as np
import random
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def perturb_text(text):
    """Slightly modifies text to test robustness (typos, case changes, slang injection)"""
    text = str(text)
    # Randomly inject Nigerian slang
    slangs = [" abeg", " sharp sharp", " o", " sha"]
    if random.random() < 0.3:
        text += random.choice(slangs)
    
    # Randomly change capitalization
    if random.random() < 0.3:
        text = text.lower()
    
    # Randomly replace some digits to mimic evasion techniques
    if random.random() < 0.3:
        text = text.replace("0", "O").replace("1", "l")
        
    return text

def main():
    print("Loading original dataset...")
    data_path = os.path.join(os.path.dirname(__file__), '..', 'database', 'data_source.csv')
    df = pd.read_csv(data_path)
    
    df = df.dropna(subset=['Message', 'Spam'])
    
    spam_df = df[df['Spam'] == 1]
    ham_df = df[df['Spam'] == 0]
    
    # Select 50 spam, 50 ham for the validation set
    test_spam = spam_df.sample(n=50, random_state=42)
    test_ham = ham_df.sample(n=50, random_state=42)
    
    validation_df = pd.concat([test_spam, test_ham])
    
    # Apply perturbation
    validation_df['Modified_Message'] = validation_df['Message'].apply(perturb_text)
    
    # Save to CSV
    out_path = os.path.join(os.path.dirname(__file__), '..', 'validation_100_set.csv')
    validation_df.to_csv(out_path, index=False)
    print(f"Saved 100 validation messages (with robustness modifications) to {out_path}")
    
    # Training Data (everything else)
    train_df = df.drop(validation_df.index)
    
    print("\nTraining Baseline Machine Learning Model (SVM + TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_df['Message'])
    y_train = train_df['Spam']
    
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    
    # Test Baseline SVM on the modified validation set
    print("Evaluating Baseline on the 100 validation messages...")
    X_test = vectorizer.transform(validation_df['Modified_Message'])
    y_test = validation_df['Spam']
    
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n=========================================")
    print("         BASELINE SVM RESULTS            ")
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
