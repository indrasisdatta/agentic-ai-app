import uuid

import streamlit as st
from langgraph_backend import chatbot, get_threads
from langchain_core.messages import HumanMessage


def reset_chat():
    thread_id = str(uuid.uuid4())
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []
    add_threads(thread_id)

def add_threads(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )
    return state.values.get("messages", [])

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = str(uuid.uuid4())

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = get_threads()

# Sidebar UI
st.sidebar.title("LangGraph Chat")

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')

for thread_id in reversed(st.session_state["chat_threads"]):
    if st.sidebar.button(thread_id, key=f"thread_{thread_id}"):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = 'user'
            else: 
                role = 'assistant'
            temp_messages.append({'role': role, 'content': msg.content})

        st.session_state['message_history'] = temp_messages

# Cnversation history UI 
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.write(message['content'])

user_input = st.chat_input("Type here")

if user_input:

    st.session_state['message_history'].append({"role": "user", "content": user_input})
    with st.chat_message('user'):
        st.text(user_input)

    CONFIG = { "configurable": {"thread_id": st.session_state['thread_id']} }

    with st.chat_message('assistant'):

        ai_message = st.write_stream(
            message_chunk.content for message_chunk, _ in chatbot.stream(
                { 'messages': [HumanMessage(content=user_input)] },
                config=CONFIG,
                stream_mode='messages'
            )
        )

    st.session_state['message_history'].append({"role": "assistant", "content": ai_message})





