import streamlit as st
import json
import os
import uuid
from langchain_core.messages import HumanMessage, AIMessage

from agent import AGENT

from utils.logging import setup_logger
setup_logger()

st.set_page_config(
    page_title="Corporate Travel Compliance Assistant",
    page_icon="✈️",
    layout="wide"
)

@st.cache_data
def load_test_cases():
    project_root = os.path.dirname(os.path.dirname(__file__))
    test_file_path = os.path.join(project_root, 'tests', 'test_queries.json')
    try:
        with open(test_file_path, 'r', encoding='utf-8') as f:
            cases = json.load(f)
            return {f"{case['id']} - {case['category']}": case['input'] for case in cases}
    except FileNotFoundError:
        return {"Could not find any test cases!": ""}

test_cases_dict = load_test_cases()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your Corporate Travel Compliance Assistant. How can I help you with your travel policies or expenses today?"}
    ]

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

with st.sidebar:
    st.title("⚙️ Control Panel")
    st.markdown("Select a test case to auto-fill the chat, or type your own query.")
    
    selected_test = st.selectbox(
        "Load Evaluation Case", 
        list(test_cases_dict.keys())
    )
    
    if st.button("Clear Conversation"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your Corporate Travel Compliance Assistant. How can I help you with your travel policies or expenses today?"}
        ]
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

st.title("✈️ Travel & Finance Assistant")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt_text = None

if selected_test in test_cases_dict:
    prompt_text = test_cases_dict[selected_test]

user_query = st.chat_input("Type your question here...")
if user_query:
    prompt_text = user_query

if prompt_text:
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    with st.chat_message("user"):
        st.markdown(prompt_text)

    with st.chat_message("assistant"):
        with st.spinner("Thinking and consulting policies..."):
            try:
                inputs = {"messages": [HumanMessage(content=prompt_text)]}
                
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                
                response = AGENT.invoke(inputs, config=config)
                
                final_message = response["messages"][-1].content
                
                st.markdown(final_message)
                
                st.session_state.messages.append({"role": "assistant", "content": final_message})
                
            except Exception as e:
                error_msg = f"**Error executing graph:** {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})