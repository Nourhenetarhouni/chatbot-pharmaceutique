import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import faiss
import pickle
from langchain_community.docstore.in_memory import InMemoryDocstore
from sentence_transformers import SentenceTransformer
from tensorflow import keras

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"



# Load vector database
def load_vector_database(index_path="vector_db.index", metadata_path="vector_db_metadata.pkl", index_to_docstore_id_path="vector_db_index_to_docstore_id.pkl"):
    index = faiss.read_index(index_path)

    with open(metadata_path, "rb") as f:
        docstore_dict = pickle.load(f)
    docstore = InMemoryDocstore(docstore_dict)

    with open(index_to_docstore_id_path, "rb") as f:
        index_to_docstore_id = pickle.load(f)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_db = FAISS(
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
        embedding_function=embeddings.embed_query
    )
    return vector_db

# Create Conversational QA Chain
def create_qa_chain(vector_db, api_key, model_name):
    llm = ChatGroq(model_name=model_name, groq_api_key=api_key)

    template = """
    You are a pharmaceutical regulation assistant.
    Before answering any question, analyze if the user provided all necessary details.

    If details are missing, ask the user a **clarifying question** instead of answering.
    Always ensure the response is **accurate and compliant with regulations**.

    Context: {context}
    Chat History: {chat_history}
    User's question: {question}

    If all details are present, provide an answer.
    If details are missing, ask a follow-up question.
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
        "question": user_query,  # ✅ Changed from "query" to "question"
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
        vector_db = load_vector_database()
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
