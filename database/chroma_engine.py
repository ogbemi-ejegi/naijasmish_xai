import os
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

# Persistent client
client = chromadb.PersistentClient(path=os.path.join(os.path.dirname(__file__), "chroma_db"))

def get_ef():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")
    return embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=api_key, model_name="models/gemini-embedding-2")

def init_db():
    try:
        collection = client.get_collection(name="scam_sms_collection_v3", embedding_function=get_ef())
        collection.query(query_texts=["dummy warmup"], n_results=1)
        print("Collection already exists and is working.")
    except Exception as e:
        print(f"Initializing new collection because: {str(e)}")
        try:
            client.delete_collection(name="scam_sms_collection_v3")
        except:
            pass
            
        collection = client.create_collection(name="scam_sms_collection_v3", embedding_function=get_ef())
        
        # Load data
        csv_path = os.path.join(os.path.dirname(__file__), 'data_source.csv')
        df = pd.read_csv(csv_path)
        
        # Filter only spam for context
        spam_df = df[df['Spam'] == 1].dropna(subset=['Message'])
        
        # Load 90 spam records to stay under the 100-request Google Free Tier limit
        spam_df = spam_df.head(90)
        
        documents = spam_df['Message'].tolist()
        ids = [f"spam_{i}" for i in range(len(documents))]
        
        # Add to collection
        collection.add(
            documents=documents,
            ids=ids
        )
        print("Collection initialized with spam data.")

def query_local_scams(preprocessed_text: str, top_k: int = 3):
    collection = client.get_collection(name="scam_sms_collection_v3", embedding_function=get_ef())
    results = collection.query(
        query_texts=[preprocessed_text],
        n_results=top_k
    )
    
    if not results['documents'] or not results['documents'][0]:
        return []
        
    return results['documents'][0]
