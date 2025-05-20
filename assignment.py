'''Citations:
1. LangChain Framework - https://github.com/langchain-ai/langchain
2. Hugging Face Transformers - https://huggingface.co/docs/transformers/index
3. Streamlit - https://docs.streamlit.io/
4. FLAN-T5 Model - https://huggingface.co/google/flan-t5-small
5. Sentence Transformers - https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
'''
import pandas as pd
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import streamlit as st
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RAGChatbot:
    """A RAG-based chatbot for answering questions about animals."""
    
    def __init__(self, data_path, model_name="google/flan-t5-small"):
        """
        Initialize the chatbot with dataset and model.
        
        Args:
            data_path (str): Path to the CSV dataset
            model_name (str): HuggingFace model name for the LLM
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
        Load and preprocess the CSV dataset.
        
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
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            
            # Create vector store
            documents = self.load_data()
            self.vector_store = FAISS.from_documents(documents, embeddings)
            
            try:
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
                
                # Move model to CPU if CUDA is not available
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
            
            # Set up RetrievalQA chain
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
        try:
            if not self.qa_chain:
                raise ValueError("RAG pipeline not initialized. Call setup_rag_pipeline first.")
            
            result = self.qa_chain({"query": question})
            answer = result["result"]
            
            # Extract the animal name from the question
            animal_query = question.lower().replace("tell me about ", "").strip()
            
            # Filter sources to only include the specific animal asked about
            sources = []
            for doc in result["source_documents"]:
                doc_text = doc.page_content.lower()
                if animal_query in doc_text:
                    sources.append(doc.page_content)
            
            response = {
                "question": question,
                "answer": answer,
                "sources": sources[:1],  # Only take the first relevant source
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
    """Main function to run the chatbot interface."""
    st.set_page_config(
        page_title="Animal Knowledge Chatbot",
        page_icon="🐾"
    )
    
    st.title("🦁 RAG Chatbot - Animal Knowledge Base")
    st.write("Ask questions about animals, and I'll answer based on my knowledge!")
    
    try:
        chatbot = init_chatbot("animals.csv")
        
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