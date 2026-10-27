from typing import Annotated

from typing_extensions import TypedDict 
from pydantic import BaseModel
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END 
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq 
import os 
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

llm_groq = ChatGroq(model="openai/gpt-oss-20b")

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    print('chat_node state', state)
    messages = state['messages']
    response = llm_groq.invoke(messages)
    return {"messages": [response]}

checkpointer = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

