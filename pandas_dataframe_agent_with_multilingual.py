import os
import logging
import json
from dotenv import load_dotenv
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import pandas as pd
import yaml
import re
from langchain.agents import Tool
from langchain_experimental.tools import PythonREPLTool
import plotly.express as px
import plotly.graph_objects as go
from langchain.agents import AgentType
from langdetect import detect

# Load API keys from config
def load_yaml(file_path):
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        return {}

# Load environment variables (for API keys)
load_dotenv()

# Ensure OpenAI API key is set
config_data = load_yaml('./config.yaml')
os.environ["OPENAI_API_KEY"] = config_data["OPENAI_API_KEY"]

# Path to the CSV file
csv_file_path = "combined_data.csv"        

# Load CSV data into a Pandas DataFrame
df = pd.read_csv(csv_file_path)

# Step 1: Initialize the LLM with better parameters
llm = ChatOpenAI(
    model="gpt-4o",  # Latest GPT-4 model
    temperature=0,  # Lower temperature for more consistent responses
    request_timeout=120,  # Increased timeout for complex queries
)

columns = df.columns
prompt = """ 
    You are a highly capable assistant for analyzing DataFrame data.
        - Always use the DataFrame loaded from the CSV file (`df`) for your analysis.
        - DO NOT generate sample data or make assumptions.
    For complex queries:
        - Identify sub-tasks within the query.
        - Use tools like Python_REPL for calculations.
        - Combine results for a coherent response.
    The DataFrame contains the following columns: {columns}.
    Note:
        - For data visualization, use only plotly express or plotly graph objects.
        - If you cannot find the information in the DataFrame, say so explicitly otherwise heavily penalized."""

# Add tools
def translate_text(params: str) -> str:
    """Translates text using OpenAI. Params should be a JSON string with 'text' and 'target_lang'."""
    try:
        inputs = json.loads(params)
        text = inputs.get("text", "")
        target_lang = inputs.get("target_lang", "en")
    except json.JSONDecodeError:
        return "Invalid input. Provide a JSON string with 'text' and 'target_lang'."

    try:
        llm = ChatOpenAI(model="gpt-4", temperature=0)
        response = llm.predict(f"Translate this text to {target_lang}: {text}")
        return response
    except Exception as e:
        return f"Error translating text: {str(e)}"

# Create the translate tool properly
translate_tool = Tool(
    name="Translate_Text",
    func=translate_text,
    description="Translates text to a specified language. Provide input as a JSON string with 'text' and 'target_lang'."
)

# Create the Python REPL tool properly
python_repl_tool = Tool(
    name="Python_REPL",
    func=PythonREPLTool().run,
    description="Useful for complex calculations. Input should be Python code as a string."
)

agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    max_iterations=10,
    allow_python_repl=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    allow_dangerous_code=True,
    early_stopping_method='force',
    tools=[python_repl_tool, translate_tool],
    prefix=prompt,
    handle_parsing_errors=True
)

# Setup logging configuration
logging.basicConfig(
    filename='pandas_agent.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def ask_dataframe_agent(user_question):
    """
    Enhanced version of ask_dataframe_agent with multilingual support
    """
    try:
        # Pre-process the question
        cleaned_question = user_question.strip().replace("\n", " ")
        
        if not validate_question(cleaned_question):
            return "Sorry, this question contains forbidden operations."
        
        # Detect the language of the question
        detected_lang = detect(cleaned_question)
        
        # Add explicit instruction to use DataFrame data
        cleaned_question = f"Using ONLY the data from the DataFrame, {cleaned_question}"
            
        # Get response
        response = agent.invoke({"input": cleaned_question})
        response = response['output']
        
        # Translate response if the detected language is not English
        if detected_lang != 'en':
            translation_params = json.dumps({
                "text": response,
                "target_lang": detected_lang
            })
            response = translate_text(translation_params)
        
        return response
        
    except Exception as e:
        logging.error(f"Error processing question: {e}")
        print(e)
        error_message = f"Sorry, I encountered an error: {str(e)}"
        
        # Translate error message if needed
        if detected_lang != 'en':
            translation_params = json.dumps({
                "text": error_message,
                "target_lang": detected_lang
            })
            error_message = translate_text(translation_params)
        
        return error_message

# Add these functions for more capabilities
def validate_question(question: str) -> bool:
    """Validate if the question is appropriate for the dataset"""
    forbidden_words = ['delete', 'drop', 'remove', 'update']
    return not any(word in question.lower() for word in forbidden_words)

def format_response(response: str) -> str:
    """Format the response for better readability"""
    # Remove any SQL or Python code blocks
    response = re.sub(r'```.*?```', '', response, flags=re.DOTALL)
    # Format numbers
    response = re.sub(r'\d+\.\d+', lambda x: f"{float(x.group()):.2f}", response)
    return response.strip()

if __name__ == "__main__":
    logging.info("Application started")
    print("Welcome to the Chatbot !!")
    print("Type 'exit' to quit.\n")
    while True:
        user_question = input("Ask a question: ")
        if user_question.lower() == "exit":
            logging.info("Application terminated by user")
            break
        response = ask_dataframe_agent(user_question)
        print(response)
