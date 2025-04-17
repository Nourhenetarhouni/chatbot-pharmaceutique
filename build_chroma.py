from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from pypdf import PdfReader
import pandas as pd
import torch
import os
import time

def load_pdf_content(file_path="CELEX_52013XC0802(04)_EN_TXT (1) GUIDELINE.pdf"):
    try:
        with open(file_path, "rb") as pdf_file:
            pdf_reader = PdfReader(pdf_file)
            return "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    except Exception as e:
        print(f"Erreur PDF : {e}")
        return ""

def load_excel_metadata(file_path="Site transfer - EU - data base.xlsx"):
    try:
        df = pd.read_excel(file_path)
        return df.set_index("Country/Region").T.to_dict()
    except Exception as e:
        print(f"Erreur Excel : {e}")
        return {}

def load_data_from_files():
    pdf_text = load_pdf_content()
    excel_data = load_excel_metadata()

    excel_docs = [
        Document(page_content=f"{region}: {details.get('Change of FP Manufacturing Site (Addition/Replacement)', '')}")
        for region, details in excel_data.items()
    ]

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    pdf_docs = [Document(page_content=t) for t in splitter.split_text(pdf_text)]

    return excel_docs + pdf_docs

def create_chromadb_streaming(documents, persist_directory="chroma_bge_m3"):
    os.makedirs(persist_directory, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3", 
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory,
        collection_name="langchain"
    )

    batch_size = 50  
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        vectordb.add_documents(batch)
        print(f" Batch {i // batch_size + 1} ajouté ({len(batch)} documents)")

    vectordb.persist()
    print(f" Base Chroma complétée avec succès dans {persist_directory}")

if __name__ == "__main__":
    print(" Chargement des documents...")
    docs = load_data_from_files()
    print(f" {len(docs)} documents à vectoriser avec BGE-m3...")
    start = time.time()
    create_chromadb_streaming(docs)
    print(f" Temps total : {round(time.time() - start, 2)} secondes")
