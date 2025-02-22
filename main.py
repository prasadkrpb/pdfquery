import os
import cassio
from langchain_community.vectorstores import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader

from dotenv import load_dotenv

load_dotenv()

# Load environment variables (Set these before running)
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Astra DB connection
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

# Initialize Groq LLM (LLaMA 2 - 7B)
llm = ChatGroq(model_name="gemma2-9b-it", groq_api_key=GROQ_API_KEY)

# Use an open-source embedding model (Hugging Face)
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load and extract text from PDF
def load_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

# Split text into chunks for vector storage
def split_text(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=800, chunk_overlap=200, length_function=len)
    return text_splitter.split_text(text)

# Store text embeddings in Astra DB
def store_embeddings(texts):
    vector_store = Cassandra(embedding=embedding, table_name="qa_mini_demo")
    vector_store.add_texts(texts)
    return VectorStoreIndexWrapper(vectorstore=vector_store)

# Query the stored embeddings using Groq LLM
def query_db(query_text, vector_store):
    return vector_store.query(query_text, llm=llm).strip()

if __name__ == "__main__":
    pdf_text = load_pdf("budget_speech.pdf")
    text_chunks = split_text(pdf_text)
    vector_store = store_embeddings(text_chunks)
    print("PDF processed and stored in Astra DB. Ready for queries.")
