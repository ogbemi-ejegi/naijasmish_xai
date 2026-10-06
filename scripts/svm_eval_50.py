import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def main():
    print("Loading original dataset...")
    data_path = os.path.join(os.path.dirname(__file__), '..', 'database', 'data_source.csv')
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['Message', 'Spam'])
    
    # Load the validation set so we know what to exclude from training
    val_path = os.path.join(os.path.dirname(__file__), '..', 'validation_100_set.csv')
    val_df_full = pd.read_csv(val_path)
    
    # Exclude the 100 validation messages from the training data using an anti-join or by matching messages
    # Since we dropped indices in baseline_eval, we'll exclude by original message text
    train_df = df[~df['Message'].isin(val_df_full['Message'])]
    
    print("Training Baseline Machine Learning Model (SVM + TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_df['Message'])
    y_train = train_df['Spam']
    
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    
    # Select the exact same 50 messages used in the Agent Evaluation
    spam_df = val_df_full[val_df_full['Spam'] == 1].head(25)
    ham_df = val_df_full[val_df_full['Spam'] == 0].head(25)
    test_50_df = pd.concat([spam_df, ham_df]).reset_index(drop=True)
    
    print("Evaluating Baseline SVM on the SAME 50 validation messages...")
    X_test = vectorizer.transform(test_50_df['Modified_Message'])
    y_test = test_50_df['Spam']
    
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n=========================================")
    print("      BASELINE SVM RESULTS (50 MSGS)     ")
    print("=========================================")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("Confusion Matrix:\n", cm)
    print("=========================================\n")

if __name__ == "__main__":
    main()
