import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def main():
    print("Loading original dataset for training...")
    data_path = os.path.join(os.path.dirname(__file__), '..', 'database', 'data_source.csv')
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['Message', 'Spam'])
    
    # Load the new dataset provided by the user
    new_data_path = os.path.join(os.path.dirname(__file__), '..', 'new_dataset.csv')
    test_df = pd.read_csv(new_data_path)
    
    # Ensure no overlap in training data (anti-join by Message)
    train_df = df[~df['Message'].isin(test_df['Message'])]
    
    print("Training Baseline Machine Learning Model (SVM + TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_df['Message'])
    y_train = train_df['Spam']
    
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    
    print("Evaluating Baseline SVM on the NEW dataset (50 messages)...")
    X_test = vectorizer.transform(test_df['Message'])
    y_test = test_df['Spam']
    
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n=========================================")
    print("   BASELINE SVM RESULTS (NEW DATASET)    ")
    print("=========================================")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("Confusion Matrix:\n", cm)
    print("=========================================\n")

if __name__ == "__main__":
    main()
