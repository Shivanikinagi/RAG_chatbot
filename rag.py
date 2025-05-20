""" Citations:
1. LangChain Framework - https://github.com/langchain-ai/langchain
2. Hugging Face Transformers - https://huggingface.co/docs/transformers/index
3. Streamlit - https://docs.streamlit.io/
4. FLAN-T5 Model - https://huggingface.co/google/flan-t5-small
5. Sentence Transformers - https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
"""


import pandas as pd
import os
import streamlit as st
import logging
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
import logging



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    import faiss
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.llms import HuggingFacePipeline
    logger.info("Successfully imported FAISS and related packages")
except ImportError as e:
    st.error("FAISS not found. Installing required packages...")
    import subprocess
    subprocess.run(["conda", "install", "-c", "conda-forge", "faiss-cpu", "-y"])
    import faiss
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.llms import HuggingFacePipeline
    logger.info("FAISS installed and imported successfully")
    
class RAGChatbot:
    """A RAG-based chatbot for answering questions about animals.
    
    Uses:
    - LangChain for RAG pipeline (https://python.langchain.com/)
    - Hugging Face Transformers for LLM (https://huggingface.co/docs/transformers)
    - FAISS for vector similarity search (https://github.com/facebookresearch/faiss)
    """
    
    def __init__(self, data_path, model_name="google/flan-t5-small"):
        """
        Initialize the chatbot with dataset and model.
        
        Args:
            data_path (str): Path to the CSV dataset
            model_name (str): HuggingFace model name for the LLM
                            (Default: google/flan-t5-small)
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at {data_path}")
            
        self.data_path = data_path
        self.model_name = model_name
        self.vector_store = None
        self.qa_chain = None
        self.responses = []
        logger.info(f"Initializing RAGChatbot with model {model_name}")

    def load_data(self):
        """
        Load and preprocess the CSV dataset using LangChain's CSVLoader.
        
        Returns:
            list: Processed document chunks
        """
        try:
            loader = CSVLoader(file_path=self.data_path)
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", ", "]
            )
            split_docs = text_splitter.split_documents(documents)
            logger.info(f"Loaded and split {len(split_docs)} document chunks")
            return split_docs
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def setup_rag_pipeline(self):
        """Set up the RAG pipeline with embeddings, vector store, and LLM."""
        try:
            # Using sentence-transformers for embeddings
            # Model: all-MiniLM-L6-v2 (https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            
            documents = self.load_data()
            self.vector_store = FAISS.from_documents(documents, embeddings)
            
            try:
                # Initialize Hugging Face model pipeline
                # Base model: google/flan-t5-small
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = model.to(device)
                
                pipe = pipeline(
                    "text2text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_length=512,
                    temperature=0.7,
                    device=0 if device == "cuda" else -1
                )
                llm = HuggingFacePipeline(pipeline=pipe)
            except Exception as model_error:
                logger.error(f"Error loading model: {str(model_error)}")
                raise        
            
            prompt_template = """Answer the question based strictly on the context.
            If the answer isn't in the context, say "I don't know."
            Don't mention animals not in the context.
            
            Context: {context}
            
            Question: {question}
            Answer: """
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt}
            )
            logger.info("RAG pipeline setup completed successfully")
        except Exception as e:
            logger.error(f"Error setting up RAG pipeline: {str(e)}")
            raise

    def ask_question(self, question: str) -> dict:
        """
        Answer a user question using the RAG pipeline.
        
        Args:
            question (str): The user's question
            
        Returns:
            dict: Contains question, answer, sources and timestamp
        """
        try:
            if not self.qa_chain:
                raise ValueError("RAG pipeline not initialized. Call setup_rag_pipeline first.")
            
            result = self.qa_chain({"query": question})
            answer = result["result"]
            
            # Filter sources to only include relevant ones
            sources = []
            question_keywords = [word.lower() for word in question.split() if len(word) > 3]
            for doc in result["source_documents"]:
                doc_text = doc.page_content.lower()
                if any(keyword in doc_text for keyword in question_keywords):
                    sources.append(doc.page_content)
            
            # If answer mentions a different animal than asked, correct it
            if "i don't know" not in answer.lower():
                question_animal = next((kw for kw in question_keywords if kw in ["lion", "tiger", "elephant"]), None)
                if question_animal:
                    answer_animals = [kw for kw in ["lion", "tiger", "elephant"] if kw in answer.lower()]
                    if answer_animals and question_animal not in answer_animals:
                        answer = "I don't know about that animal, but I found information about other animals."
                        sources = []  # Clear sources since they're not relevant
            
            response = {
                "question": question,
                "answer": answer,
                "sources": sources,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            self.responses.append(response)
            logger.info(f"Successfully answered question: {question[:50]}...")
            return response
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return {
                "question": question,
                "answer": f"Sorry, I encountered an error: {str(e)}",
                "sources": [],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

    def save_responses(self, output_file="chatbot_responses.txt"):
        """
        Save all question-answer pairs to a text file.
        
        Args:
            output_file (str): Path to the output file
        """
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                for response in self.responses:
                    f.write(f"Question: {response['question']}\n")
                    f.write(f"Answer: {response['answer']}\n")
                    if response['sources']:
                        f.write("Sources:\n")
                        for i, source in enumerate(response['sources'], 1):
                            f.write(f"{i}. {source}\n")
                    f.write(f"Timestamp: {response['timestamp']}\n")
                    f.write("-" * 80 + "\n")
            logger.info(f"Responses saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving responses: {str(e)}")
            raise

@st.cache_resource
def init_chatbot(data_path: str) -> RAGChatbot:
    """Initialize and cache the chatbot instance."""
    return RAGChatbot(data_path=data_path)

def main():
    """Main function to run the Streamlit chatbot interface."""
    # Get absolute path to the data file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(current_dir, "animal_data.csv")
    
    # Debug logging for file path
    logger.info(f"Looking for data file at: {data_file}")
    logger.info(f"File exists: {os.path.exists(data_file)}")

    st.set_page_config(
        page_title="Animal Knowledge Chatbot",
        page_icon="🐾"
    )
    
    st.title("🦁 RAG Chatbot - Animal Knowledge Base")
    st.write("Ask questions about animals, and I'll answer based on my knowledge!")
    
    try:
        chatbot = init_chatbot(data_file)
        
        if "rag_setup" not in st.session_state:
            with st.spinner("Initializing chatbot..."):
                chatbot.setup_rag_pipeline()
                st.session_state.rag_setup = True
        
        question = st.text_input("🤔 What would you like to know about animals?")
        
        if st.button("Ask Question 🔍"):
            if question:
                with st.spinner("Thinking..."):
                    response = chatbot.ask_question(question)
                    
                    st.write("---")
                    st.write(f"**Q:** {response['question']}")
                    st.write(f"**A:** {response['answer']}")
                    
                    if response['sources']:
                        st.write("\n**Sources:**")
                        for i, source in enumerate(response['sources'], 1):
                            st.write(f"{i}. {source}")
                    
                    chatbot.save_responses()
            else:
                st.warning("⚠️ Please enter a question first!")
                
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        logger.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()