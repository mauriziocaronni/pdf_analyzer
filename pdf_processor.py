# File: pdf_processor.py
import os
import time
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

import json
from pathlib import Path
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from datetime import datetime
import PyPDF2


from utils import get_text, get_prompt, clean_response, json_to_excel, init_mistral_llm

# qui
# Imposta la variabile d'ambiente per risolvere il conflitto di OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
#os.environ["OMP_NUM_THREADS"] = "1"

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# Ottieni il valore di PATH
source_path = os.getenv('source_dir')
destination_path = os.getenv('destination_dir')
split_pdf_path = os.getenv('split_pdf_dir')
excel_file_path = "output.xlsx"

# crea un'istanza di DocumentConverter
converter = DocumentConverter()



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
        self.streamlit_status = st.empty()
        self.progress_bar = st.progress(0)
        self.step_status = st.empty()
        
    def update_status(self, message, progress_value=None):
        """
        Update processing status using the callback if available.
        
        Args:
            message: Status message to display
            progress_value: Optional value (0.0 to 1.0) to update progress bar
        """
        if self.status_callback:
            self.status_callback(message)
        
        # Update Streamlit elements
        self.streamlit_status.text(message)
        if progress_value is not None:
            self.progress_bar.progress(progress_value)
            
        # Always print to terminal for logging
        print(message)
        
        # Force Streamlit to update by adding a small delay
        # This helps ensure UI updates are visible during processing
        time.sleep(0.1)
        # Use st.rerun() instead of deprecated st.experimental_rerun()
        try:
            st.rerun()
        except:
            # Fallback for older versions of Streamlit
            pass
    
# step 1

    def split_pdf(self, input_pdf_path, filename , output_dir):

        # Apri il PDF di input in modalità lettura binaria
        with open(input_pdf_path+"/"+filename, "rb") as file:
            reader = PyPDF2.PdfReader(file)

            num_pages = len(reader.pages)
            start = 0
            part_num = 1
            n_split = 1

            # Suddividi il PDF in blocchi da N pagine per volta
            while start < num_pages:
                end = min(start + n_split, num_pages)  # ci fermiamo al massimo se esauriamo le pagine

                # Crea un nuovo writer per il PDF parziale
                writer = PyPDF2.PdfWriter()

                # Aggiungi le pagine dalla 'start' alla 'end - 1'
                for page in range(start, end):
                    writer.add_page(reader.pages[page])

                # Salva il PDF risultante
                output_file_path = Path(output_dir) / f"{Path(filename).stem}_{part_num:04}.pdf"
                         
                with open(output_file_path, "wb") as output_file:
                        writer.write(output_file)
                # Aggiorna gli indici per la prossima iterazione
                start = end
                part_num += 1

# step 2 e 3
    def convert_to_md(self, split_pdf_path, filename, destination_path):
        split_pdf_path = Path(split_pdf_path) / filename
        md_path = Path(destination_path) / os.path.basename(filename)
        markdown_path = md_path.with_suffix('.md')
        try:
            result = converter.convert(str(split_pdf_path))
            markdown_content = result.document.export_to_markdown()
            with open(markdown_path, 'w', encoding='utf-8') as md_file:
                md_file.write(markdown_content)            
        except Exception as e:
            print(f"Errore durante la conversione di {filename}: {e}")

    def extract_information_from_markdown(self, destination_path, filename, progress_callback=None):
        """
        Extract information from a markdown file and update Excel.
        
        Args:
            destination_path: Path to markdown files directory
            filename: Name of the file to process
            progress_callback: Optional callback function for progress updates
        """
        md_path = Path(destination_path) / os.path.basename(filename)
        markdown_path = md_path.with_suffix('.md')
        
        # Update progress if callback provided
        if progress_callback:
            progress_callback(f"Extracting information from {filename}")
            
        input_text = get_text(markdown_path)
        complete_prompt = get_prompt(input_text)
        
        try:
            # Call model
            if progress_callback:
                progress_callback(f"Generating response for {filename}")
                
            # Call the Mistral model using LangChain's WatsonxLLM interface
            progress_callback(f"Calling Mistral model for {filename}...")
            
            # Use the proper mistral_llm instance from the class
            response_text = self.mistral_llm.invoke(complete_prompt)

            # Extract JSON from response_text
            clean_text = clean_response(response_text)
            
            # Parse the cleaned JSON string
            try:
                json_data = json.loads(clean_text)
                
                # If we got valid JSON data, add the file path
                if json_data and isinstance(json_data, list):
                    for item in json_data:
                        item['file_path'] = filename
                else:
                    # If we got empty or invalid data, create a placeholder for logging
                    if progress_callback:
                        progress_callback(f"No valid data extracted from {filename}")
                    json_data = []
            except json.JSONDecodeError:
                if progress_callback:
                    progress_callback(f"Failed to parse JSON from {filename}")
                json_data = []
     
            # Convert JSON to Excel
            json_to_excel(json_data, excel_file_path)
            
            if progress_callback:
                progress_callback(f"Updated Excel file with data from {filename}")
                
            return True
            
        except json.JSONDecodeError as e:
            error_msg = f"Error decoding JSON from {filename}: {e}"
            print(error_msg)
            if progress_callback:
                progress_callback(error_msg)
            return False
         


# process 
    def step1_split_pdf(self, pdf_path):
        """
        STEP 1: Split the PDF into individual pages.
        """
        # Initialize progress for step 1
        self.progress_bar.progress(0.0)
        self.step_status.text("Step 1/3: Splitting PDF into individual pages")
        self.update_status(f"📊 Splitting PDF: {pdf_path}", progress_value=0.2)
        
        try:
            # Split the PDF
            self.split_pdf(os.path.dirname(pdf_path), os.path.basename(pdf_path), split_pdf_path)
            
            # Update progress after step completion
            self.update_status(f"✅ Step 1 completed: PDF split into individual pages", progress_value=1.0)
            return True
            
        except Exception as e:
            self.update_status(f"❌ Error in PDF splitting step: {str(e)}", progress_value=0.5)
            return False
    
    def step2_convert_to_markdown(self):
        """
        STEP 2: Convert split PDFs to markdown.
        """
        # Initialize progress for step 2
        self.progress_bar.progress(0.0)
        self.step_status.text("Step 2/3: Converting PDF pages to markdown")
        self.update_status(f"📊 Converting PDF pages to markdown format", progress_value=0.1)
        
        try:
            # Get list of split PDF files
            file_list = sorted([f for f in os.listdir(split_pdf_path) if f.endswith('.pdf')])
            total_files = len(file_list)
            
            if total_files == 0:
                self.update_status("❌ No PDF files found in split directory. Run Step 1 first.", progress_value=0.0)
                return False
            
            # Process each file with progress updates
            for idx, filename in enumerate(file_list):
                file_progress = idx / total_files
                progress_value = 0.1 + (file_progress * 0.8)  # Scale to 0.1-0.9 range
                
                self.update_status(f"📄 Converting file {idx+1}/{total_files}: {filename}", 
                                   progress_value=progress_value)
                
                try:
                    self.convert_to_md(split_pdf_path, filename, destination_path)
                except Exception as e:
                    self.update_status(f"⚠️ Warning: Error converting {filename}: {str(e)}", 
                                      progress_value=progress_value)
            
            # Update progress after step completion
            self.update_status(f"✅ Step 2 completed: {total_files} files converted to markdown", 
                              progress_value=1.0)
            return True
            
        except Exception as e:
            self.update_status(f"❌ Error in markdown conversion step: {str(e)}", progress_value=0.5)
            return False
    
    def step3_extract_information(self):
        """
        STEP 3: Extract information from markdown files.
        """
        # Initialize progress for step 3
        self.progress_bar.progress(0.0)
        self.step_status.text("Step 3/3: Extracting information from markdown files")
        self.update_status(f"📊 Extracting information from markdown files", progress_value=0.1)
        
        try:
            # Get list of markdown files from destination_path
            md_files = sorted([
                f 
                for f in os.listdir(destination_path) 
                if f.endswith('.md')
            ])
            total_md_files = len(md_files)
            
            if total_md_files == 0:
                self.update_status("❌ No markdown files found. Run Step 2 first.", progress_value=0.0)
                return False
                
            # Create a progress update callback that includes UI updates
            def progress_update(message):
                self.update_status(message)
            
            # Process each markdown file with progress updates
            successful_extractions = 0
            for idx, md_filename in enumerate(md_files):
                file_progress = idx / total_md_files
                progress_value = 0.1 + (file_progress * 0.8)  # Scale to 0.1-0.9 range
                
                self.update_status(f"📄 Extracting from file {idx+1}/{total_md_files}: {md_filename}", 
                                   progress_value=progress_value)
                
                try:
                    if self.extract_information_from_markdown(
                        destination_path, 
                        md_filename, 
                        progress_callback=progress_update
                    ):
                        successful_extractions += 1
                except Exception as e:
                    self.update_status(f"⚠️ Warning: Error extracting from {md_filename}: {str(e)}", 
                                     progress_value=progress_value)
            
            # Update progress after step completion
            self.update_status(
                f"✅ Step 3 completed: Extracted information from {successful_extractions}/{total_md_files} files", 
                progress_value=1.0
            )
            return True
            
        except Exception as e:
            self.update_status(f"❌ Error in information extraction step: {str(e)}", progress_value=0.5)
            return False
    
    def process_pdf(self, pdf_path):
        """
        Process a PDF file by running all three steps sequentially.
        This is kept for backward compatibility.
        """
        self.update_status(f"📄 Starting processing of {os.path.basename(pdf_path)}", progress_value=0.0)
        self.step_status.text("Running all processing steps sequentially")
        
        # Run Step 1: Split PDF
        if not self.step1_split_pdf(pdf_path):
            return False
            
        # Run Step 2: Convert to markdown
        if not self.step2_convert_to_markdown():
            return False
            
        # Run Step 3: Extract information
        if not self.step3_extract_information():
            return False
            
        self.step_status.text(f"✅ All 3 steps completed successfully!")
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