# File: utils.py
import os
import json
from dotenv import load_dotenv

from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from langchain_ibm import WatsonxLLM
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
import pandas as pd

def get_credentials():
    """
    Get WatsonX.ai credentials from environment variables.
    """
    load_dotenv()
    
    return {
        "apikey": os.getenv("WATSONX_API_KEY"),
        "url": os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
        "project_id": os.getenv("WATSONX_PROJECT_ID")
    }

def init_mistral_llm(verbose=False):
    """
    Initialize a WatsonxLLM instance with Mistral Large model.
    """
    credentials = get_credentials()
    
    # Check if credentials are available
    if not credentials["apikey"] or not credentials["project_id"]:
        if verbose:
            print("❌ Missing WatsonX credentials. Set WATSONX_API_KEY and WATSONX_PROJECT_ID environment variables.")
        return None
    
    watsonx_apikey = credentials["apikey"]
    watsonx_url = credentials["url"]
    watsonx_project_id = credentials["project_id"]
    
   # Parametri per il modello Mistral Large
    model_params  = {
    GenTextParamsMetaNames.DECODING_METHOD: "greedy",
    GenTextParamsMetaNames.MAX_NEW_TOKENS: 13000,
    GenTextParamsMetaNames.MIN_NEW_TOKENS: 1,
    GenTextParamsMetaNames.TEMPERATURE: 0,
    GenTextParamsMetaNames.TOP_K: 50,
    GenTextParamsMetaNames.TOP_P: 1,
    }


    # Create WatsonxLLM instance
    try:
        mistral_llm = WatsonxLLM(
        model_id="mistralai/mistral-large",
        url=watsonx_url,
        apikey=watsonx_apikey,
        project_id=watsonx_project_id,
        params=model_params,
    )
        if verbose:
            print("✅ Mistral Large model initialized successfully.")
        return mistral_llm
    except Exception as e:
        if verbose:
            print(f"❌ Error initializing Mistral Large model: {str(e)}")
        return None

def save_uploaded_file(uploaded_file, upload_dir="uploads"):
    """
    Save an uploaded file to the specified directory.
    Returns the path to the saved file.
    """
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
        
    file_path = os.path.join(upload_dir, uploaded_file.name)
    
    # Save the uploaded file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    return file_path



# funzioni di utilita
def clean_response(response_text):
    """
    Clean the response text from the model to extract just the JSON.
    """
    # If empty response (for testing)
    if not response_text:
        return "[]"
        
    # First find JSON brackets in the response
    start_idx = response_text.find('[')
    end_idx = response_text.rfind(']')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        # Extract just the JSON part
        json_part = response_text[start_idx:end_idx+1]
    else:
        # Try standard cleaning if direct JSON extraction fails
        clean_text = response_text.replace("```", "")
        clean_text = clean_text.replace("json", "")
        clean_text = clean_text.replace("\\_", "")
        clean_text = clean_text.replace("<text>", "")
        clean_text = clean_text.replace("<instruction>", "")
        clean_text = clean_text.replace("</instruction>", "")
        clean_text = clean_text.replace("<output>", "")
        clean_text = clean_text.replace("</output>", "")
        clean_text = clean_text.replace("###END###", "")
        json_part = clean_text
    
    try:
        # Parse to ensure valid JSON and filter out examples
        json_data = json.loads(json_part)
        # Filter out items with "Esempio" in the description
        filtered_json_data = [item for item in json_data if "Esempio" not in item.get("Descrizione", "")]
        clean_text = json.dumps(filtered_json_data, ensure_ascii=False)
        print(f"Cleaned JSON: {clean_text}")
        return clean_text
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print(f"Original text: {response_text}")
        print(f"Attempted JSON part: {json_part}")
        # Return empty array as fallback
        return "[]"

def json_to_excel(json_data, excel_file_path):
    # Convert JSON to DataFrame
    df = pd.DataFrame(json_data)
    
    if os.path.exists(excel_file_path):
        # Append to existing Excel file
        with pd.ExcelWriter(excel_file_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
    else:
        # Write to a new Excel file
        df.to_excel(excel_file_path, index=False)

def get_text(file_path=None):
    with open(file_path, 'r') as file:
        file_content = file.read()
    input_text = file_content
    return input_text

# fine


# parte AI

def get_prompt(input_text):

    # Get the complete prompt by replacing variables
    complete_prompt = f"""
<instruction>
Sei un impiegato della Regione e ti è stato assegnato il compito di estrarre alcune informazioni da un testo contenente deliberazioni regionali.

Estrai dal seguente testo tutte le deliberazioni regionali. 
Non confondere le deliberazioni con le leggi regionali. Deve essere presente la parola "Delibera" o "Deliberazione" per essere considerata tale.
Non inventare o inferire NESSUNA informazione non presente nel testo.

Per ogni deliberazione trovata, estrai solo:
- Numero: [solo se esplicitamente menzionato]
- Data: [solo se esplicitamente menzionata]
- Descrizione: [solo se esplicitamente menzionata]
- Pagina: [solo se esplicitamente menzionata]

Formatta l'output come JSON strutturato con i campi:
```json
[
  {{
    "Numero": "valore trovato",
    "Data": "valore trovato",
    "Descrizione": "valore trovato",
    "Pagina": "valore trovato"
  }}
]

ISTRUZIONI IMPORTANTI:

Se non riesci a trovare un valore per il campo "Numero", NON generare alcun JSON.
Genera un JSON SOLO se il campo "Numero" è presente e ha un valore.
Non aggiungere alcuna spiegazione o commento al JSON.

Esempio di output desiderato da non includere nel JSON:
[
  {{
    "Numero": "123",
    "Data": "2023-01-01",
    "Descrizione": "Esempio Approvazione del bilancio",
    "Pagina": "5"
  }},
  {{
    "Numero": "456",
    "Data": "2023-02-15",
    "Descrizione": "Esempio Nomina del nuovo dirigente",
    "Pagina": "10"
  }}
]

</instruction> <text> ```{input_text}``` </text> ```
        """
    return complete_prompt

# fine