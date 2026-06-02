

AI PDF Chatbot (RAG System)
A Retrieval-Augmented Generation (RAG) application that allows you to chat with your PDF documents using Flask, LangChain, and Google Generative AI.

🚀 Overview
This application processes PDF documents, indexes their content into a vector database (FAISS), and leverages an LLM to provide context-aware answers to user queries based on the provided documents.

🛠 Tech Stack
Framework: Flask

Orchestration: LangChain

Vector Store: FAISS

Embeddings: HuggingFace sentence-transformers

LLM: Google Generative AI

PDF Processing: pypdf

📦 Installation
Clone the repository:

Bash
git clone [YOUR_REPOSITORY_URL]
cd indian-h
Create a virtual environment:

Bash
python -m venv venv
# Windows:
venv\Scripts\activate
Install dependencies:

Bash
pip install -r requirements.txt
Set your API Key:
Create a .env file in the root directory and add your Google API Key:

Plaintext
GOOGLE_API_KEY=your_actual_api_key_here
⚙️ Running the Application
Start the Flask development server:

Bash
python app.py
The API will be available at http://127.0.0.1:8000
