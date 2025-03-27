# PDF Analyzer with Mistral & LangChain

This application allows you to upload PDF documents, process them using WatsonX.ai's Mistral Large language model, and interactively ask questions about the content.

## Features

- Upload PDF documents
- Real-time processing status updates
- Ask questions about the document content
- Generate document summaries
- Uses Retrieval Augmented Generation (RAG) for accurate answers

## Technologies Used

- **IBM WatsonX.ai**: Provides access to the Mistral Large language model
- **LangChain**: Framework for document processing and RAG implementation
- **Streamlit**: Interactive web interface
- **FAISS**: Vector store for efficient similarity search
- **Sentence Transformers**: For document embeddings

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your WatsonX.ai credentials (see `.env.sample`)
4. Run the application:
   ```
   streamlit run app.py
   ```

## WatsonX.ai Credentials

You need the following credentials from IBM WatsonX.ai:

- API Key
- Project ID

These can be entered directly in the application or stored in a `.env` file.

## Usage

1. Upload a PDF document
2. Click "Process PDF" to analyze the document
3. Ask questions or generate a summary
4. View processing status in real-time

## Directory Structure

- `app.py`: Main Streamlit application
- `pdf_processor.py`: PDF processing and RAG implementation
- `utils.py`: Utility functions for WatsonX.ai integration
- `uploads/`: Directory for uploaded PDF files

## License

MIT
