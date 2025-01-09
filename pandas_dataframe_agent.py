import os
import logging
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
from openai import OpenAI
from guardrails import Guard

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
csv_file_path = "data_with_address.csv"

# csv_file_path = "combined_data.csv"     
# csv_file_path = "data.csv"
# csv_file_path = "modelling_data.csv"

# Load CSV data into a Pandas DataFrame
df = pd.read_csv(csv_file_path)
# print(df.head(5))
# print(df.columns)
# print(df.info())

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

agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    max_iterations=10,
    allow_python_repl=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    allow_dangerous_code=True,
    early_stopping_method='force',
    tools=[
        Tool(
            name="Python_REPL",
            func=PythonREPLTool().run,
            description="Useful for complex calculations. Input should be Python code as a string."
        )
    ],
    prefix= prompt,
    handle_parsing_errors=True
)

# Setup logging configuration
logging.basicConfig(
    filename='pandas_agent.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize the OpenAI client
client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"]
)

def ask_dataframe_agent(user_question):
    """
    Enhanced version of ask_dataframe_agent with Guardrails validation
    """
    try:
        # Pre-process the question
        cleaned_question = user_question.strip().replace("\n", " ")
        
        if not validate_question(cleaned_question):
            return "Sorry, this question contains forbidden operations."
        
        # Add explicit instruction to use DataFrame data
        cleaned_question = f"Using ONLY the data from the DataFrame, {cleaned_question}"
            
        # Get raw response from the agent
        raw_response = agent.invoke({"input": cleaned_question})
        response_text = raw_response['output']

        try:
            # Create guard with schema
            guard = Guard.fetch_guard("./guardrails_schema.yaml")
            # Validate response
            validated_response = guard.parse(response_text)
            response = validated_response.answer
        except Exception as e:
            # If guardrails validation fails, use the raw response
            logging.warning(f"Guardrails validation failed: {e}")
            response = response_text
                
        return response
        
    except Exception as e:
        logging.error(f"Error processing question: {e}")
        print(e)
        return f"Sorry, I encountered an error: {str(e)}"

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

# Add OpenAI Moderation Function
def check_moderation(text):
    """
    Checks if the input text violates OpenAI's moderation guidelines.
    """
    try:
        response = client.moderations.create(input=text)
        results = response.results[0]
        
        return {
            "flagged": results.flagged,
            "categories": {
                cat: flagged 
                for cat, flagged in results.categories.model_dump().items() 
                if flagged
            }
        }
    except Exception as e:
        print(f"Error during moderation check: {e}")
        return {"flagged": False, "categories": {}}

if __name__ == "__main__":
    logging.info("Application started")
    print("Welcome to the Chatbot !!")
    print("Type 'exit' to quit.\n")
    while True:
        user_question = input("Ask a question: ")

        # Pre-process the question
        cleaned_question = user_question.strip().replace("\n", " ")
        
        # Moderation check for user input
        moderation_result = check_moderation(cleaned_question.lower())
        if moderation_result["flagged"]:
            print("Chatbot: Your input contains inappropriate content. Please try again.")
            print(f"Flagged categories: {', '.join(moderation_result['categories'].keys())}")
            continue

        if user_question.lower() == "exit":
            logging.info("Application terminated by user")
            break
        response = ask_dataframe_agent(user_question)
        print(response)

