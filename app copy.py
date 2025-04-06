import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# Load the vector database (assuming it's already created and saved)
def load_vector_database(index_path="vector_db.index", metadata_path="vector_db_metadata.pkl", index_to_docstore_id_path="vector_db_index_to_docstore_id.pkl"):
    import faiss
    import pickle
    from langchain.docstore.in_memory import InMemoryDocstore
    from sentence_transformers import SentenceTransformer

    # Load the FAISS index
    index = faiss.read_index(index_path)

    # Load the docstore (metadata)
    with open(metadata_path, "rb") as f:
        docstore_dict = pickle.load(f)
    docstore = InMemoryDocstore(docstore_dict)

    # Load the index_to_docstore_id mapping
    with open(index_to_docstore_id_path, "rb") as f:
        index_to_docstore_id = pickle.load(f)

    # Load the embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Reconstruct the vector database
    vector_db = FAISS(
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
        embedding_function=embeddings.embed_query
    )
    print("Vector database loaded successfully.")
    return vector_db

# Create QA chain with Groq LLM
def create_qa_chain(vector_db, api_key, model_name):
    groq_llm = ChatGroq(api_key=api_key, model_name=model_name)

    template = """You are an AI assistant that provides medical regulatory information.
    Use the following context to answer the question at the end.
    Context: {context}

    Question: {question}
    Answer:
    """

    PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=groq_llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )
    return qa_chain

def ask_chatbot(qa_chain, product_type, country, type_of_change):
    question = f"What are the requirements for {type_of_change} for {product_type} in {country}?"
    
    # Retrieve context from the vector database
    context_docs = qa_chain.retriever.get_relevant_documents(question)
    context = " ".join([doc.page_content for doc in context_docs])
    print("Context:", context)
    print("Question:", question)
    # Ensure correct key naming when invoking the chain
    response = qa_chain.invoke({"query": question, "context": context})

    return response.get("result", "No answer found.")


# Main Streamlit app
def main():
    st.title("Regulatory Information Chatbot")

    # Single text input for the query
    user_query = st.text_input("Enter your query:")

    # Button to submit the query
    if st.button("Submit"):
        # Load the vector database
        vector_db = load_vector_database()

        # Create QA chain with Groq LLM
        api_key = "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"
        model_name = "llama3-8b-8192"
        qa_chain = create_qa_chain(vector_db, api_key, model_name)

        # Get the answer from the chatbot
        response = qa_chain.invoke({"query": user_query})
        answer = response.get("result", "No answer found.")
        st.write("Answer:")
        st.write(answer)

if __name__ == "__main__":
    main()
