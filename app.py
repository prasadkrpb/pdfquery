import streamlit as st
#import cassio
from main import load_pdf, split_text, store_embeddings, query_db

st.title("PDF Querying with RAG (Astra DB + Groq LLaMA 2)")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("File uploaded successfully!")

    # Process PDF
    pdf_text = load_pdf("temp.pdf")
    text_chunks = split_text(pdf_text)
    vector_store = store_embeddings(text_chunks)

    st.session_state["vector_store"] = vector_store
    st.session_state["processed"] = True
    st.write("PDF processed and ready for querying!")

if "processed" in st.session_state and st.session_state["processed"]:
    query_text = st.text_input("Enter your query:")
    if query_text:
        answer = query_db(query_text, st.session_state["vector_store"])
        st.write("**Answer:**", answer)
