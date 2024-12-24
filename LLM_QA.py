import os
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import yaml
from langchain.embeddings import OpenAIEmbeddings

# Load environment variables (for API keys)
load_dotenv()

# Load API keys from config
def load_yaml(file_path):
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        return {}

config_data = load_yaml('./config.yaml')
os.environ["OPENAI_API_KEY"] = config_data["OPENAI_API_KEY"]

# Load CSV file
csv_file_path = "combined_data.csv"  # Replace with your CSV file path
loader = CSVLoader(file_path=csv_file_path, encoding="utf-8")

# Create a VectorstoreIndex from the CSV data
index_creator = VectorstoreIndexCreator(
    embedding=OpenAIEmbeddings(),
    vectorstore_kwargs={}
)
docsearch = index_creator.from_loaders([loader])

# Initialize the OpenAI Chat model
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

# Create a Retrieval-based QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=docsearch.vectorstore.as_retriever(),
    return_source_documents=True
)

# Function to ask questions
def ask_question(question):
    response = qa_chain({"query": question})
    answer = response["result"]
    sources = response["source_documents"]
    
    print("\nAnswer:", answer)
    print("\nSources:")
    for source in sources:
        print(source.page_content)

# Example usage
question = "How many total orders are there?"  # Replace with your query
ask_question(question)
