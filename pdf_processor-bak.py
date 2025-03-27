# File: pdf_processor.py
import os
import time
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

class PdfProcessor:
    def __init__(self, mistral_llm, status_callback=None):
        """
        Initialize a PDF processor with Mistral LLM.
        
        Args:
            mistral_llm: A WatsonxLLM instance for Mistral Large
            status_callback: A function to call with status updates
        """
        self.mistral_llm = mistral_llm
        self.status_callback = status_callback
        self.vectorstore = None
        self.qa_chain = None
        
    def update_status(self, message):
        """
        Update processing status using the callback if available.
        """
        if self.status_callback:
            self.status_callback(message)
        # Always print to console for logging
        print(message)
    
    def process_pdf(self, pdf_path):
        """
        Process a PDF file and create a vector store for RAG.
        """
        self.update_status(f"📄 Loading PDF: {os.path.basename(pdf_path)}")
        
        # 1. Load and split the PDF
        try:
            loader = PyPDFLoader(pdf_path)
            time.sleep(0.5)  # Brief pause for UI updates
            pages = loader.load_and_split()
            self.update_status(f"📊 Loaded {len(pages)} pages")
        except Exception as e:
            self.update_status(f"❌ Error loading PDF: {str(e)}")
            return False
        
        # 2. Split into chunks
        self.update_status("🔪 Splitting document into chunks")
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200
            )
            time.sleep(0.5)  # Brief pause for UI updates
            chunks = text_splitter.split_documents(pages)
            self.update_status(f"🧩 Created {len(chunks)} text chunks")
        except Exception as e:
            self.update_status(f"❌ Error splitting document: {str(e)}")
            return False
            
        # 3. Create embeddings
        self.update_status("🔤 Generating embeddings")
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
            time.sleep(0.5)  # Brief pause for UI updates
        except Exception as e:
            self.update_status(f"❌ Error initializing embeddings: {str(e)}")
            return False
            
        # 4. Create the vector store
        self.update_status("🗄️ Creating vector store")
        try:
            self.vectorstore = FAISS.from_documents(chunks, embeddings)
            time.sleep(0.5)  # Brief pause for UI updates
            self.update_status("✅ Vector store created successfully")
        except Exception as e:
            self.update_status(f"❌ Error creating vector store: {str(e)}")
            return False
            
        # 5. Create QA chain
        self.update_status("⚙️ Setting up QA chain")
        try:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.mistral_llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever()
            )
            time.sleep(0.5)  # Brief pause for UI updates
            self.update_status("🎉 PDF processing complete!")
        except Exception as e:
            self.update_status(f"❌ Error setting up QA chain: {str(e)}")
            return False
            
        return True
    
    def query_document(self, question):
        """
        Query the processed document with a question.
        """
        if not self.qa_chain:
            self.update_status("❌ No document has been processed yet.")
            return None
            
        self.update_status(f"🔍 Querying: '{question}'")
        try:
            response = self.qa_chain.run(question)
            self.update_status("✅ Response received")
            return response
        except Exception as e:
            self.update_status(f"❌ Error during query: {str(e)}")
            return None
    
    def generate_summary(self):
        """
        Generate a summary of the document.
        """
        if not self.vectorstore:
            self.update_status("❌ No document has been processed yet.")
            return None
            
        self.update_status("📝 Generating document summary")
        try:
            query = "Summarize this document in 3-5 paragraphs, highlighting the key points and main topics covered."
            retrieved_docs = self.vectorstore.similarity_search(query, k=4)
            
            context = " ".join([doc.page_content for doc in retrieved_docs])
            prompt = f"Context: {context}\n\nBased only on the provided context, {query}"
            
            summary = self.mistral_llm.invoke(prompt)
            self.update_status("✅ Summary generated")
            return summary
        except Exception as e:
            self.update_status(f"❌ Error generating summary: {str(e)}")
            return None