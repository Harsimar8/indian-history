from flask import Flask, render_template, request, jsonify
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

app = Flask(__name__)

# ==========================================
# CONFIGURATION
# ==========================================
FAISS_INDEX_PATH = "faiss_index" 
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PDF_FILE_PATH = "indian-history.pdf" 

# ==========================================
# INITIALIZE RAG (Using FAISS)
# ==========================================
print("Initializing Knowledge Base with FAISS... Please wait.")
try:
    # 1. Load and Split PDF
    loader = PyPDFLoader(PDF_FILE_PATH)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    docs = text_splitter.split_documents(pages)
    
    # 2. Create Embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # 3. Create FAISS Vector Store
    vector_db = FAISS.from_documents(docs, embeddings)
    
    # 4. Save the index locally
    vector_db.save_local(FAISS_INDEX_PATH)
    print("FAISS Knowledge Base Ready! (Running in Pure Retrieval Mode)")
except Exception as e:
    print(f"Error loading PDF: {e}")
    vector_db = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_query = request.json.get("message")
    if vector_db is None:
        return jsonify({"answer": "PDF not loaded."})
    
    # RAG Retrieval using FAISS
    # This finds the top 3 most relevant chunks of text from your PDF
    docs = vector_db.similarity_search(user_query, k=3)
    
    if not docs:
        return jsonify({"answer": "No relevant information found in the PDF."})

    # Instead of calling an AI, we just combine the results from the PDF
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # We return the exact text found in the PDF as the answer
    answer = f"According to the document:\n\n{context}"
    
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True, port=5000)

