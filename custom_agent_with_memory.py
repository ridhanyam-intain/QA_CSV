import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.tools import PythonAstREPLTool
from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_core.messages import ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnableMap, RunnableLambda

from operator import itemgetter
import yaml
import os
from dotenv import load_dotenv


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
# Load the dataframe
df = pd.read_csv(csv_file_path)

# Define the Python execution tool
tool = PythonAstREPLTool(locals={"df": df})

# Create the tool parser
parser = JsonOutputKeyToolsParser(key_name="query", first_tool_only=True)

# Define the system prompt
system = f"""You have access to a pandas dataframe `df`. \
Here is the output of `df.head().to_markdown()`:

```markdown
{df.head().to_markdown()}
```

Given a user question, write the Python code to answer it. \
Return ONLY the valid Python code and nothing else. \
Don't assume you have access to any libraries other than built-in Python ones and pandas."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "{question}"),
    MessagesPlaceholder("chat_history", optional=True),
])

# Define the function to extract chat history
def _get_chat_history(ai_msg, tool_output):
    """Parse the chain output up to this point into a list of chat history messages to insert in the prompt."""
    tool_call_id = ai_msg.additional_kwargs["tool_calls"][0]["id"]
    tool_msg = ToolMessage(tool_call_id=tool_call_id, content=str(tool_output))
    return [ai_msg, tool_msg]

# Define the chain
chain = RunnableMap({
    "ai_msg": lambda x: prompt.invoke({"question": x["question"]}),
    "tool_output": lambda x: tool.invoke(x["ai_msg"]),
    "chat_history": lambda x: _get_chat_history(x["ai_msg"], x["tool_output"]),
    "response": lambda x: StrOutputParser().invoke(prompt.invoke({"chat_history": x["chat_history"], "question": x["question"]}))
}).pick(["tool_output", "response"])

# Execute the chain
result = chain.invoke({"question": "What's the correlation between age and fare"})

# Output the result
print(result)
