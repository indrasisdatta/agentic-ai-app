from typing import Annotated

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from typing_extensions import TypedDict 
from pydantic import BaseModel
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END 
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq 
import os 
from dotenv import load_dotenv
import requests

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

llm_groq = ChatGroq(model="openai/gpt-oss-20b")

search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()

tools = [search_tool, get_stock_price, calculator]
llm_with_tools = llm_groq.bind_tools(tools)



class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    print('chat_node state', state)
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

pool = ConnectionPool(
    conninfo=os.environ["DATABASE_URL"],
    min_size=2,
    max_size=10,
    kwargs={
        "autocommit": True,
        "prepare_threshold": 0,
        "row_factory": dict_row,
    },
    open=False
)

pool.open() 

# checkpointer = InMemorySaver()

checkpointer = PostgresSaver(pool)

# Creates the checkpoint tables / applies migrations.
# Run as a controlled initialization or migration step.
checkpointer.setup()

tool_node = ToolNode(tools)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tools)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)

graph.add_edge("tools","chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

def get_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
       all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)