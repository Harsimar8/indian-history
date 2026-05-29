from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
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
UPLOAD_FOLDER = "uploaded_pdfs"
ALLOWED_EXTENSIONS = {"pdf"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FAISS_INDEX_PATH, exist_ok=True)

print("Initializing Knowledge Base... Please wait.")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vector_db = None

# Load existing index if available
try:
    if os.path.isdir(FAISS_INDEX_PATH) and os.listdir(FAISS_INDEX_PATH):
        vector_db = FAISS.load_local(FAISS_INDEX_PATH, embeddings)
        print("Loaded existing FAISS index.")
    else:
        print("No existing FAISS index found. Upload a PDF to start.")
except Exception as e:
    print(f"Error loading existing FAISS index: {e}")
    vector_db = None


def allowed_file(filename):
    return "." in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_vector_store_from_pdf(filepath):
    loader = PyPDFLoader(filepath)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    docs = text_splitter.split_documents(pages)
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(FAISS_INDEX_PATH)
    return vector_store


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/status')
def status():
    if vector_db is not None:
        return jsonify({"ready": True, "message": "PDF is loaded and ready."})
    return jsonify({"ready": False, "message": "No PDF loaded yet. Upload a PDF to start."})


@app.route('/upload', methods=['POST'])
def upload():
    global vector_db
    if 'pdf' not in request.files:
        return jsonify({"success": False, "error": "No file part in the request."}), 400

    file = request.files['pdf']
    if file.filename == '':
        return jsonify({"success": False, "error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Only PDF files are allowed."}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    try:
        vector_db = create_vector_store_from_pdf(save_path)
        return jsonify({"success": True, "message": "PDF uploaded and indexed successfully."})
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to index PDF: {e}"}), 500


@app.route('/ask', methods=['POST'])
def ask():
    if vector_db is None:
        return jsonify({"answer": "PDF not loaded. Please upload a PDF first."})

    user_query = request.json.get("message")
    if not user_query:
        return jsonify({"answer": "Please ask a valid question."})

    docs = vector_db.similarity_search(user_query, k=3)
    if not docs:
        return jsonify({"answer": "No relevant information found in the uploaded PDF."})

    context = "\n\n".join([doc.page_content for doc in docs])
    answer = f"According to the document:\n\n{context}"
    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(debug=True, port=5000)

