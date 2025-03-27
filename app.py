# File: app.py
import streamlit as st
import os
import time
from utils import init_llm, save_uploaded_file
from pdf_processor import PdfProcessor

# Page configuration
st.set_page_config(
    page_title="PDF Analyzer with AI",
    page_icon="📄",
    layout="wide"
)

# Initialize session state variables
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
    
if 'processor' not in st.session_state:
    st.session_state.processor = None
    
if 'status_messages' not in st.session_state:
    st.session_state.status_messages = []
    
if 'model_provider' not in st.session_state:
    st.session_state.model_provider = "openai"  # Default to OpenAI

# Function to append messages to status log
def update_status(message):
    """
    Add a new status message to the session state
    and ensure the status container gets updated.
    """
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.status_messages.append(f"[{timestamp}] {message}")

# Main application
st.title("PDF Analyzer with AI")

# Sidebar setup
with st.sidebar:
    st.header("About")
    st.markdown("""
        This application uses:
        - **AI** for document understanding
        - **LangChain** for document processing
        - **Streamlit** for the user interface
    """)
    
    st.header("Model Configuration")
    
    # Model provider selection
    model_provider = st.radio(
        "Select AI Model Provider",
        ["OpenAI", "WatsonX/Mistral"],
        index=0 if st.session_state.model_provider == "openai" else 1,
        help="Choose which AI model provider to use for document analysis"
    )
    
    # Update session state if the model provider changed
    selected_provider = "openai" if model_provider == "OpenAI" else "watsonx"
    if st.session_state.model_provider != selected_provider:
        st.session_state.model_provider = selected_provider
        if st.session_state.processor is not None:
            st.session_state.processor = None
            update_status(f"🔄 Switching to {model_provider} model")
    
    # Check if credentials exist
    missing_credentials = False
    
    if st.session_state.model_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        missing_credentials = True
        st.error("⚠️ OpenAI API credentials not found")
        
        # Input fields for credentials
        with st.form("openai_credentials_form"):
            api_key = st.text_input("OpenAI API Key", type="password")
            submitted = st.form_submit_button("Save Credentials")
            
            if submitted and api_key:
                # In a real app, you might want to store these securely
                os.environ["OPENAI_API_KEY"] = api_key
                st.success("✅ OpenAI credentials saved for this session")
                missing_credentials = False
                st.experimental_rerun()
                
    elif st.session_state.model_provider == "watsonx" and (not os.getenv("WATSONX_API_KEY") or not os.getenv("WATSONX_PROJECT_ID")):
        missing_credentials = True
        st.error("⚠️ WatsonX credentials not found")
        
        # Input fields for credentials
        with st.form("watsonx_credentials_form"):
            api_key = st.text_input("WatsonX API Key", type="password")
            project_id = st.text_input("WatsonX Project ID")
            submitted = st.form_submit_button("Save Credentials")
            
            if submitted and api_key and project_id:
                # In a real app, you might want to store these securely
                os.environ["WATSONX_API_KEY"] = api_key
                os.environ["WATSONX_PROJECT_ID"] = project_id
                st.success("✅ WatsonX credentials saved for this session")
                missing_credentials = False
                st.experimental_rerun()

# Display status messages
status_container = st.container()

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Upload and Process PDF Document")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file and not missing_credentials:
        # Initialize LLM if needed
        if st.session_state.processor is None:
            model_name = "OpenAI GPT-4o" if st.session_state.model_provider == "openai" else "WatsonX Mistral"
            update_status(f"🔄 Initializing {model_name} model...")
            llm = init_llm(model_provider=st.session_state.model_provider)
            
            if llm:
                st.session_state.processor = PdfProcessor(llm, update_status)
                update_status(f"✅ {model_name} model initialized successfully")
            else:
                update_status(f"❌ Failed to initialize {model_name} model")
                st.error(f"Unable to initialize {model_name} model. Check credentials.")
        
        # Save the file
        if 'pdf_path' not in st.session_state:
            pdf_path = save_uploaded_file(uploaded_file)
            st.session_state.pdf_path = pdf_path
            update_status(f"📁 Saved file to {pdf_path}")
        
        # Create three separate buttons for each processing step
        st.subheader("Choose Processing Steps")
        
        # Step 1 Button
        if st.button("Step 1: Split PDF"):
            with st.spinner("Splitting PDF into pages..."):
                if st.session_state.processor.step1_split_pdf(st.session_state.pdf_path):
                    update_status("✅ Step 1 completed successfully")
                    st.success("PDF successfully split into individual pages")
                else:
                    st.error("Error splitting PDF. Check the status log for details.")
        
        # Step 2 Button        
        if st.button("Step 2: Convert to Markdown"):
            with st.spinner("Converting pages to markdown..."):
                if st.session_state.processor.step2_convert_to_markdown():
                    update_status("✅ Step 2 completed successfully")
                    st.success("PDF pages successfully converted to markdown")
                else:
                    st.error("Error converting to markdown. Check the status log for details.")
        
        # Step 3 Button
        if st.button("Step 3: Extract Information"):
            with st.spinner("Extracting information from markdown files..."):
                if st.session_state.processor.step3_extract_information():
                    st.session_state.pdf_processed = True
                    update_status("✅ Step 3 completed successfully")
                    st.success("Information successfully extracted from markdown files")
                    st.balloons()
                else:
                    st.error("Error extracting information. Check the status log for details.")
        
        # Full Process Button
        if st.button("Run All Steps"):
            with st.spinner("Processing PDF through all steps..."):
                if st.session_state.processor.process_pdf(st.session_state.pdf_path):
                    st.session_state.pdf_processed = True
                    update_status("✅ All steps completed successfully")
                    st.success("PDF fully processed!")
                    st.balloons()
                else:
                    st.error("Error processing the PDF. Check the status log for details.")

#with col2:
#    if st.session_state.pdf_processed:
#        st.header("Ask Questions")
#        
#        # Summary button
#        if st.button("Generate Summary"):
#            with st.spinner("Generating summary..."):
#                summary = st.session_state.processor.generate_summary()
#                if summary:
#                    st.subheader("Document Summary")
#                    st.write(summary)
#                else:
#                    st.error("Failed to generate summary.")
#        
#        # Question input and answer button
#        question = st.text_area("Ask a question about the document")
#        if question and st.button("Ask"):
#            with st.spinner("Thinking..."):
#                answer = st.session_state.processor.query_document(question)
#                if answer:
#                    st.subheader("Answer")
#                    st.write(answer)
#                else:
#                    st.error("Failed to get an answer.")
    else:
        st.info("👈 Upload and process a PDF to start asking questions")

# Status display
with status_container:
    st.header("Processing Status")
    status_area = st.empty()
    
    # Display all status messages as a formatted block
    if st.session_state.status_messages:
        status_text = "\n".join(st.session_state.status_messages)
        status_area.code(status_text, language=None)
    else:
        status_area.info("Ready to process documents...")

# Footer
st.markdown("""---
*Built with LangChain and Streamlit*""")