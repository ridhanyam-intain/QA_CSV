import os
import logging
from dotenv import load_dotenv
from langchain_experimental.agents import create_csv_agent
from langchain_community.chat_models import ChatOpenAI
from deep_translator import GoogleTranslator
import yaml
from functools import lru_cache
import re

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
csv_file_path = "combined_data.csv"  # Replace with your CSV file path

# Step 1: Initialize the LLM with better parameters
llm = ChatOpenAI(
    model="gpt-4o",  # Latest GPT-4 model
    temperature=0.1,  # Lower temperature for more consistent responses
    request_timeout=120,  # Increased timeout for complex queries
    max_retries=3  # Add retries for reliability
)

# Step 2: Create an enhanced CSV Agent
agent = create_csv_agent(
    llm,
    csv_file_path,
    verbose=True,
    agent_type="openai-tools",  
    handle_parsing_errors=True,
    max_iterations=5,  
    early_stopping_method="generate",  
    allow_dangerous_code=True,
    prefix="""You are a helpful assistant analyzing delivery data. 
    When providing numerical answers, round to 2 decimal places.
    Always provide clear, concise responses.
    If analyzing trends or patterns, focus on actionable insights.
    For location-based queries, consider both the restaurant and delivery areas."""
)

# Setup logging configuration
logging.basicConfig(
    filename='csv_agent.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@lru_cache(maxsize=100)
def cached_query(query: str) -> str:
    """
    Cached version of agent.run() to avoid repeated API calls for the same query.
    """
    logging.info(f"Cache miss for query: {query}")
    return agent.run(query)

def ask_csv_agent(user_question, target_lang="en"):
    """
    Enhanced version of ask_csv_agent with better error handling and response processing
    """
    try:
        # Pre-process the question
        cleaned_question = user_question.strip().replace("\n", " ")
        
        # Add context if needed
        if "average" in cleaned_question.lower():
            cleaned_question += " (round to 2 decimal places)"
            
        # Get response
        response = cached_query(cleaned_question)
        
        # Post-process response
        response = response.replace("```", "").replace("python", "")
        response = response.strip()
        
        # Translate if needed
        if target_lang != "en":
            try:
                translator = GoogleTranslator(source='en', target=target_lang)
                response = translator.translate(text=response)
            except Exception as e:
                logging.error(f"Translation error: {e}")
                return "Translation error occurred"
                
        return response
        
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

# Example usage
if __name__ == "__main__":
    logging.info("Application started")
    
    print("Welcome to the Chatbot !!")
    print("Type 'exit' to quit.\n")
    while True:
        user_question = input("Ask a question: ")
        if user_question.lower() == "exit":
            logging.info("Application terminated by user")
            break
        ask_csv_agent(user_question)
