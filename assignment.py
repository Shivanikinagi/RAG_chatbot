import pandas as pd
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import streamlit as st
import os

# Task 1: Data Loading
def load_data(file_path):
    """
    Loads a CSV dataset to be used as the knowledge base for the RAG chatbot.
    
    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        documents: Loaded documents from the CSV file or None if an error occurs.
    """
    try:
        loader = CSVLoader(file_path=file_path, encoding='utf-8')
        documents = loader.load()
        return documents
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Task 2: Set Up RAG Pipeline with LangChain
def setup_rag_pipeline(documents):
    """
    Sets up the RAG pipeline using LangChain with FAISS vector store and HuggingFace model.
    
    Args:
        documents: List of documents loaded from the dataset.
    
    Returns:
        RetrievalQA: Configured RAG pipeline for question answering or None if setup fails.
    """
    try:
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Create FAISS vector store
        vector_store = FAISS.from_documents(documents, embeddings)
        
        # Initialize the language model using transformers pipeline
        model_name = "distilgpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=False)
        
        # Create a text-generation pipeline
        text_generation_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            temperature=0.7,
            truncation=True
        )
        
        # Wrap the pipeline in a LangChain-compatible LLM
        llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
        
        # Define prompt template
        prompt_template = """Use the following pieces of context to answer the question. If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Answer: """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create RAG pipeline
        rag_pipeline = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        return rag_pipeline
    except Exception as e:
        print(f"Error setting up RAG pipeline: {e}")
        return None

# Task 3: Build Chatbot
def run_chatbot(rag_pipeline, question):
    """
    Runs the chatbot to answer a given question using the RAG pipeline.
    
    Args:
        rag_pipeline: Configured RetrievalQA pipeline.
        question (str): User's question.
    
    Returns:
        dict: Contains the answer and source documents or an error message.
    """
    try:
        response = rag_pipeline({"query": question})
        # Extract the answer part from the response
        answer = response["result"].split("Answer: ")[-1].strip()
        # Extract source documents
        sources = [doc.page_content for doc in response["source_documents"]]
        return {"answer": answer, "sources": sources}
    except Exception as e:
        return {"answer": f"Error processing question: {e}", "sources": []}

# Streamlit App for Chatbot
def streamlit_app(rag_pipeline):
    """
    Runs a Streamlit-based chatbot interface with the specified design.
    
    Args:
        rag_pipeline: Configured RetrievalQA pipeline.
    """
    # Set page title with emoji
    st.title("RAG Chatbot - Animal Knowledge Base 🐾")
    st.write("Ask questions about animals, and I'll answer based on my knowledge!")

    # Placeholder question
    st.markdown("🎯 *What would you like to know about animals?*")

    # Input box for user question
    user_input = st.text_input("Ask Question", value="tell me about lion", placeholder="tell me about lion")

    # Ask button
    if st.button("Ask Question"):
        if user_input:
            # Get chatbot response
            result = run_chatbot(rag_pipeline, user_input)
            answer = result["answer"]
            sources = result["sources"]

            # Display question and answer
            st.markdown(f"**Q:** {user_input}")
            st.markdown(f"**A:** {answer}")

            # Display sources
            if sources:
                st.markdown("**Sources:**")
                for i, source in enumerate(sources, 1):
                    st.markdown(f"{i}. {source}")

# Main function to execute the tasks
def main():
    """
    Main function to load data, set up RAG pipeline, and run the chatbot.
    """
    # Sample dataset file (animal knowledge base)
    sample_csv = "animal_data.csv"
    
    # Create a sample dataset if it doesn't exist
    if not os.path.exists(sample_csv):
        sample_data = pd.DataFrame({
            "name": [
                "Lion",
                "Elephant",
                "Giraffe"
            ],
            "description": [
                "The Lion is a large carnivorous mammal known for its strength and social behavior, living in prides, habitat: Savannas and grasslands of Africa, diet: Carnivore, eats antelopes, zebras, and other mammals",
                "The Elephant is the largest land animal, known for its intelligence and long trunk, habitat: Forests, savannas, and grasslands of Africa and Asia, diet: Herbivore, eats grasses, leaves, and bark",
                "The Giraffe is the tallest land animal with a long neck, habitat: Savannas and grasslands of Africa, diet: Herbivore, eats leaves from tall trees, especially acacias"
            ]
        })
        sample_data.to_csv(sample_csv, index=False)
    
    # Task 1: Load data
    documents = load_data(sample_csv)
    if not documents:
        print("Failed to load data. Exiting.")
        return
    
    # Task 2: Set up RAG pipeline
    rag_pipeline = setup_rag_pipeline(documents)
    if not rag_pipeline:
        print("Failed to set up RAG pipeline. Exiting.")
        return
    
    # Run Streamlit app
    streamlit_app(rag_pipeline)

    # Save sample questions and responses
    sample_questions = [
        "tell me about lion",
        "tell me about elephant",
        "tell me about giraffe",
        "tell me about penguins"  # Test case for unknown animal
    ]
    
    with open("chatbot_responses.txt", "w") as f:
        for question in sample_questions:
            result = run_chatbot(rag_pipeline, question)
            answer = result["answer"]
            sources = result["sources"]
            f.write(f"Question: {question}\nResponse: {answer}\n")
            f.write("Sources:\n")
            for i, source in enumerate(sources, 1):
                f.write(f"{i}. {source}\n")
            f.write("\n")

if __name__ == "__main__":
    main()