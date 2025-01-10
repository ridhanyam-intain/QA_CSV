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
import guardrails as gd
from guardrails.hub import RestrictToTopic
from guardrails.errors import ValidationError

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
csv_file_path = "combined_data_v6.csv"     
# Load CSV data into a Pandas DataFrame
df = pd.read_csv(csv_file_path)

# Step 1: Initialize the LLM with better parameters
llm = ChatOpenAI(
    model="gpt-4",  # Latest GPT-4 model
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
    Enhanced version with Guardrails
    """
    try:
        # Pre-process the question
        cleaned_question = user_question.strip().replace("\n", " ")
        
        # Add explicit instruction to use DataFrame data
        cleaned_question_with_instructions = f"Using ONLY the data from the DataFrame, {cleaned_question}"

        # Get response from the agent
        raw_response = agent.invoke({"input": cleaned_question_with_instructions})
        llm_output = raw_response['output']  # This is the actual LLM output
        # print("LLM Output: ", llm_output)
        
        # Validate topic using Guardrails
        guard = gd.Guard.for_string(
            validators=[
                RestrictToTopic(
                    valid_topics=["finance", "delivery", "restaurant"],
                    invalid_topics=["phone", "tablet", "sports", "politics"],
                    device=-1,
                    llm_callable="gpt-3.5-turbo",
                    disable_classifier=False,
                    disable_llm=False,
                    on_fail="exception",
                )
            ]
        )
        
        # Validate the LLM output's topic
        try:
            guard.parse(llm_output=llm_output)
            print("Topic validation passed")
        except ValidationError as e:
            print("Topic validation failed")
            logging.warning(f"Topic validation failed: {e}")
            return "Sorry, your question appears to be outside the scope of our supported topics. Please ask questions related to finance, delivery, or restaurant data."
        
        return llm_output
        
    except Exception as e:
        logging.error(f"Error processing question: {e}")
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
