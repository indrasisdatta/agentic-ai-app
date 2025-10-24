import os
from dotenv import load_dotenv
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun 
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults 
from langchain_groq import ChatGroq 
from langchain_core.messages import HumanMessage
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from typing import Annotated
from langgraph.graph.message import add_messages

from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# Arxiv 
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=2, doc_content_chars=250)
arxiv = ArxivQueryRun(api_wrapper = api_wrapper_arxiv)

# Wikipedia
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars=500)
wiki = WikipediaQueryRun(api_wrapper = api_wrapper_wiki)

# Tavily tool for web search 
tavily = TavilySearchResults()

tools = [arxiv, wiki, tavily]

llm = ChatGroq(model="openai/gpt-oss-20b")
llm_with_tools = llm.bind_tools(tools)

# State schema

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Node definition 
def tool_calling_llm(state: State):
    return {"messages": llm_with_tools.invoke(state["messages"])}

memory = MemorySaver()

# Build Graph 
builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm", tools_condition)
# builder.add_edge("tools", END)
builder.add_edge("tools", "tool_calling_llm")

graph = builder.compile()
# graph = builder.compile(checkpointer=memory)

# display(Image(graph.get_graph().draw_mermaid_png()))

# config = {
#     "configurable": {"thread_id": 2}
# }

