# File: utils.py
import os
import json
from dotenv import load_dotenv

# Import WatsonX dependencies conditionally
try:
    from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
    from langchain_ibm import WatsonxLLM
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
    WATSONX_AVAILABLE = True
except ImportError:
    WATSONX_AVAILABLE = False

# Import OpenAI dependencies
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

import pandas as pd

def get_credentials():
    """
    Get credentials from environment variables.
    """
    load_dotenv()
    
    return {
        "watsonx_apikey": os.getenv("WATSONX_API_KEY"),
        "watsonx_url": os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
        "watsonx_project_id": os.getenv("WATSONX_PROJECT_ID"),
        "openai_apikey": os.getenv("OPENAI_API_KEY")
    }

def init_llm(model_provider="watsonx", verbose=True):  # Set verbose=True by default
    """
    Initialize an LLM instance based on the specified provider.
    
    Args:
        model_provider (str): The provider to use ('watsonx' or 'openai')
        verbose (bool): Whether to print verbose output
        
    Returns:
        LLM instance or None if initialization fails
    """
    print(f"Initializing LLM with provider: {model_provider}")
    credentials = get_credentials()
    
    # Print available credentials (masked for security)
    print(f"WatsonX API Key available: {bool(credentials['watsonx_apikey'])}")
    print(f"WatsonX Project ID available: {bool(credentials['watsonx_project_id'])}")
    print(f"OpenAI API Key available: {bool(credentials['openai_apikey'])}")
    
    if model_provider.lower() == "watsonx":
        if not WATSONX_AVAILABLE:
            error_msg = "❌ WatsonX dependencies not available. Install with 'pip install langchain-ibm ibm-watson-machine-learning'"
            print(error_msg)
            return None
            
        # Check if WatsonX credentials are available
        if not credentials["watsonx_apikey"] or not credentials["watsonx_project_id"]:
            error_msg = "❌ Missing WatsonX credentials. Set WATSONX_API_KEY and WATSONX_PROJECT_ID environment variables."
            print(error_msg)
            return None
        
        # Parametri per il modello Mistral Large
        model_params = {
            GenTextParamsMetaNames.DECODING_METHOD: "greedy",
            GenTextParamsMetaNames.MAX_NEW_TOKENS: 13000,
            GenTextParamsMetaNames.MIN_NEW_TOKENS: 1,
            GenTextParamsMetaNames.TEMPERATURE: 0,
            GenTextParamsMetaNames.TOP_K: 50,
            GenTextParamsMetaNames.TOP_P: 1,
        }

        # Create WatsonxLLM instance
        try:
            print("Attempting to initialize WatsonX/Mistral...")
            llm = WatsonxLLM(
                model_id="mistralai/mistral-large",
                url=credentials["watsonx_url"],
                apikey=credentials["watsonx_apikey"],
                project_id=credentials["watsonx_project_id"],
                params=model_params,
            )
            print("✅ Mistral Large model initialized successfully.")
            return llm
        except Exception as e:
            error_msg = f"❌ Error initializing Mistral Large model: {str(e)}"
            print(error_msg)
            # Print stack trace for debugging
            import traceback
            print(traceback.format_exc())
            return None
    
    elif model_provider.lower() == "openai":
        # Direct OpenAI approach without using LangChain
        try:
            # Check for OpenAI API key
            if not credentials["openai_apikey"]:
                print("❌ Missing OpenAI API key. Please set OPENAI_API_KEY environment variable.")
                return None
                
            # Import OpenAI library directly
            try:
                import openai
            except ImportError:
                print("❌ OpenAI package not installed. Run 'pip install openai'")
                return None
                
            # Create a simple OpenAI client
            api_key = credentials["openai_apikey"]
            print(f"Using OpenAI API key (masked): {api_key[:4]}...{api_key[-4:]}")
            
            # Create a custom LLM class that mimics the LangChain interface
            class SimpleOpenAILLM:
                def __init__(self, api_key):
                    self.client = openai.OpenAI(api_key=api_key)
                    
                def invoke(self, prompt):
                    try:
                        response = self.client.chat.completions.create(
                            model="gpt-4o",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0
                        )
                        return response.choices[0].message.content
                    except Exception as e:
                        print(f"❌ Error invoking OpenAI: {e}")
                        raise e
            
            llm = SimpleOpenAILLM(api_key)
            print("✅ OpenAI GPT-4o model initialized successfully")
            return llm
            
        except Exception as e:
            print(f"❌ Error initializing OpenAI: {e}")
            import traceback
            print(traceback.format_exc())
            return None
    
    else:
        if verbose:
            print(f"❌ Invalid model provider: {model_provider}. Choose 'watsonx' or 'openai'.")
        return None

# For backward compatibility
def init_mistral_llm(verbose=False):
    """
    Initialize a WatsonxLLM instance with Mistral Large model (legacy function).
    """
    return init_llm(model_provider="watsonx", verbose=verbose)

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
    # Read prompt from external file
    try:
        with open('prompt_01.txt', 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        
        # Replace the {input_text} placeholder with actual input
        complete_prompt = prompt_template.format(input_text=input_text)
    except Exception as e:
        print(f"Error reading prompt file: {e}")
        # Fallback to a basic prompt if file can't be read
        complete_prompt = f"<instruction>Extract deliberation data from text.</instruction> <text>{input_text}</text>"
    return complete_prompt

# fine