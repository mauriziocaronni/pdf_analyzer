import os
import json
from pathlib import Path
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from datetime import datetime
import PyPDF2


# Imposta la variabile d'ambiente per risolvere il conflitto di OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
#os.environ["OMP_NUM_THREADS"] = "1"

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# Ottieni il valore di PATH
source_path = os.getenv('source_dir')
destination_path = os.getenv('destination_dir')
split_pdf_path = os.getenv('split_pdf_dir')

# crea un'istanza di DocumentConverter
converter = DocumentConverter()

def split_pdf(input_pdf_path, output_dir):
    for file_name in os.listdir(input_pdf_path):
     if file_name.lower().endswith('.pdf'):
        pdf_path = Path(input_pdf_path) / file_name
        print(f"\nProcessing file: {pdf_path}")
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
        
                num_pages = len(reader.pages)
                start = 0
                part_num = 1
        
                while start < num_pages:
                    end = min(start + 1, num_pages)
                    writer = PyPDF2.PdfWriter()
            
                    for page in range(start, end):
                        writer.add_page(reader.pages[page])
                    
                    output_file_path = Path(output_dir) / f"{Path(file_name).stem}_{part_num:04}.pdf"
                         
                    with open(output_file_path, "wb") as output_file:
                        writer.write(output_file)
            
                    start = end
                    part_num += 1
        except Exception as e:
            print(f"Errore durante la divisione del pdf  {file_name}: {e}")

def process_file(args):
    file_name, source_path, destination_path = args
    pdf_path = Path(source_path) / file_name
    md_path = Path(destination_path) / file_name

    print(f"\nProcessing file: {pdf_path}")

    try:
        print(f"Converting {file_name} to markdown...", datetime.now())
        result = converter.convert(str(pdf_path))
        markdown_content = result.document.export_to_markdown()
        markdown_path = md_path.with_suffix('.md')
        print(f"Saving markdown to: {markdown_path}")
        with open(markdown_path, 'w', encoding='utf-8') as md_file:
            md_file.write(markdown_content)
        print(f"Markdown saved in: {markdown_path}", datetime.now())

        info = extract_information_from_markdown(markdown_content)

    except Exception as e:
        print(f"Errore durante l'elaborazione di {file_name}: {e}")

def extract_information_from_markdown(markdown_content):
    # Implementa l'estrazione delle informazioni dal contenuto markdown
    pass

def extract_information(source_path, destination_path):
    file_list = sorted(os.listdir(source_path))  # Ordina i file in ordine alfabetico
    args = [(file_name, source_path, destination_path) for file_name in file_list if file_name.lower().endswith('.pdf')]
    for arg in args:
        process_file(arg)

def main ():

    print("start", datetime.now())
    print("source path: ", source_path)
    print("split pdf path: ", split_pdf_path)
    print("destination path: ", destination_path)

    split_pdf(source_path, split_pdf_path)

    print("reading files in " + split_pdf_path)
    extract_information(split_pdf_path, destination_path)
    print("done!", datetime.now())
    

if __name__ == "__main__":
    main()