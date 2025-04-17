import os
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from pypdf import PdfReader
import pandas as pd


import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


# Load vector database
def load_chroma_vector_db(persist_directory="chroma_bge_m3"):
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    vector_db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="langchain"  
    )
    return vector_db


# Create Conversational QA Chain
def create_qa_chain(vector_db, api_key, model_name):
    llm = ChatGroq(model_name=model_name, groq_api_key=api_key)

    template = """
    You are a pharmaceutical regulation assistant. Your goal is to give regulatory answers that are accurate and helpful, even when user input is incomplete.

    Follow this reasoning process:

    1. Analyze the user’s question carefully.
    2. If key information is missing (e.g. product name, country, indication), try to guess what the user might mean based on the question.
    3. Provide a general but useful answer, using examples or known regulations when possible.
    4. If the answer depends heavily on missing info, mention this briefly and invite the user to clarify.
    5. Always be helpful and avoid asking multiple follow-up questions unless strictly necessary.
    


    Context: {context}
    Chat History: {chat_history}
    User's question: {question}

    Give a helpful, clear, and regulation-compliant answer. Do not refuse to answer unless absolutely required.
    """


    prompt = PromptTemplate(input_variables=["context", "chat_history", "question"], template=template)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db.as_retriever(),
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
    )

    return qa_chain

# Function to interact with chatbot
def ask_chatbot(qa_chain, user_query):
    # Retrieve chat history from session state
    chat_history = [(msg["role"], msg["content"]) for msg in st.session_state.chat_history]

    response = qa_chain.invoke({
        "question": user_query,  
        "chat_history": chat_history
    })

    return response.get("answer", "No answer found.")



# Fonction pour afficher l'historique du chat
import streamlit as st

# Fonction pour afficher l'historique du chat
def display_chat_history():
    st.markdown("<h3 style='text-align: center;'>💬 Discussion</h3>", unsafe_allow_html=True)
    
    if "chat_history" in st.session_state:
        for message in st.session_state.chat_history:
            role = message["role"]
            content = message["content"]

            # Appliquer un style différent pour l'utilisateur et le chatbot
            if role == "user":
                st.markdown(
                    f"<div style='text-align: right; background-color: #dcf8c6; padding: 10px; border-radius: 10px; margin: 5px 0; width: 60%; float: right;'>"
                    f" {content}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='text-align: left; background-color: #f1f0f0; padding: 10px; border-radius: 10px; margin: 5px 0; width: 60%; float: left;'>"
                    f" {content}</div>",
                    unsafe_allow_html=True,
                )

    st.markdown("<div style='clear: both;'></div>", unsafe_allow_html=True)  # Nettoyage

# Fonction qui enregistre l'entrée utilisateur
def on_input_change():
    st.session_state.user_input_text = st.session_state.user_input

# Fonction principale
def main():
    st.title("🤖 Chatbot de Réglementation Pharmaceutique")

    # Initialisation de l'historique
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Initialisation de l'entrée utilisateur
    if "user_input_text" not in st.session_state:
        st.session_state.user_input_text = ""

    # Affichage de l'historique du chat
    display_chat_history()

    # Champ de saisie (stocke dans `st.session_state.user_input_text`)
    st.text_input("Tapez votre message...", key="user_input", on_change=on_input_change)

    # Bouton d'envoi
    if st.button("Envoyer") and st.session_state.user_input_text.strip():
        vector_db = load_chroma_vector_db("chroma_bge_m3")
        api_key = "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"
        model_name = "llama3-8b-8192"
        qa_chain = create_qa_chain(vector_db, api_key, model_name)

        # Ajouter la question à l'historique
        st.session_state.chat_history.append({"role": "user", "content": st.session_state.user_input_text})

        # Affichage du message "Chatbot en train de répondre..."
        with st.spinner("Chatbot réfléchit... 🤔"):
            answer = ask_chatbot(qa_chain, st.session_state.user_input_text)

        # Ajouter la réponse du chatbot à l'historique
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

        # Effacer l'entrée utilisateur
        st.session_state.user_input_text = ""

        # Rafraîchir l'affichage
        st.rerun()

    # Option pour relancer la conversation
    if st.button("🗑️ Nouvelle Conversation"):
        st.session_state.chat_history.clear()
        st.session_state.user_input_text = ""
        st.rerun()

if __name__ == "__main__":
    main()
