from langgraph.graph import StateGraph, START
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq 
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langgraph.types import interrupt, Command
from dotenv import load_dotenv
import requests
import os

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="openai/gpt-oss-20b")

@tool
def get_stock_price(symbol: str) -> dict:   
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    )
    r = requests.get(url)
    return r.json()

@tool
def get_purchase_tool(symbol: str, quantity: int) -> dict:  
    """
    Simulate purchasing a given quantity of a stock symbol.

    HUMAN-IN-THE-LOOP:
    Before confirming the purchase, this tool will interrupt
    and wait for a human decision ("yes" / anything else).
    """
    # This pauses the graph and returns control to the caller 
    decision = interrupt(f"Approve buying ${quantity} shares of ${symbol}")

    if isinstance(decision, str) and decision.lower() == 'yes':
        return {
            'status': 'success',
            'message': f'Purchase order placed for ${quantity} shares of ${symbol}',
            'symbol': symbol,
            'quantity': quantity
        }
    else:
        return {
            'status': 'cancelled',
            'message': f'Purchase order of ${quantity} shares of ${symbol} was declined by human',
            'symbol': symbol,
            'quantity': quantity
        }

    
tools = [get_stock_price, get_purchase_tool]

llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    response = llm_with_tools.invoke(state['messages'])
    return {"messages": [response]}


tool_node = ToolNode(tools)

memory = MemorySaver()

graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)

graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=memory)


if __name__ == "__main__":

    thread_id = "thread-1"
    CONFIG = {'configurable': {'thread_id': thread_id}}

    while True: 
        user_input = input("You: ")
        if user_input.lower().strip() in {'exit', 'quit'}:
            print('Goodbye')
            break 

        state = {'messages': [HumanMessage(content=user_input)]}

        result = chatbot.invoke(
            state,
            config=CONFIG
        )

        """
        Graph reaches interrupt and pauses 
        {
            "__interrupt__": [
                # Interrupt information
            ]
        }
        """

        interrupts = result.get('__interrupt__', [])


        """
        1. Command(resume="yes") tells LangGraph to resume the interrupted execution.
        2. LangGraph identifies the paused execution using the same thread_id.
        3. It resumes from the interrupted task.
        4. The interrupt() expression now returns "yes".
        5. The purchase tool continues executing.
        """
        if interrupts:
            prompt_to_human = interrupts[0].value 
            print(f"HITL: {prompt_to_human}")
            decision = input('Your decision: ').strip().lower() 

            # Resume graph with the human decision yes/no 
            result = chatbot.invoke(
                Command(resume=decision),
                config=CONFIG
            )

        last_message = result['messages'][-1]
        print(f"Bot: {last_message.content}\n")