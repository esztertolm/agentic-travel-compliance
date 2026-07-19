import streamlit as st
import json
import os
import uuid
from langchain_core.messages import HumanMessage
from utils.logging import setup_logger
setup_logger()

from agent import AGENT


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

    run_test_btn = st.button("Run Selected Test Case")
    
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

if run_test_btn and selected_test in test_cases_dict:
    prompt_text = test_cases_dict[selected_test]

user_query = st.chat_input("Type your question here...")
if user_query:
    prompt_text = user_query

if prompt_text:
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    with st.chat_message("user"):
        st.markdown(prompt_text)

    with st.chat_message("assistant"):
        try:
            inputs = {"messages": [HumanMessage(content=prompt_text)]}
            
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            # Agent status indicator
            status = st.status("Agent is analyzing the request...", expanded=True)
            
            final_message = ""
            
            # Stream the graph execution to show intermediate steps
            for event in AGENT.stream(inputs, config=config):
                for node_name, state_update in event.items():
                    status.write(f"**Step completed:** `{node_name}`")
                    
                    if node_name == 'action_node':
                        tool_name = "Unknown Tool"
                        if "messages" in state_update and state_update["messages"]:
                            last_msg = state_update["messages"][-1]
                            # If it's a ToolMessage, LangChain stores the tool name in the .name attribute
                            if hasattr(last_msg, "name") and last_msg.name:
                                tool_name = last_msg.name
                        
                        with status.expander(f"🛠️ Tool executed: `{tool_name}`"):
                            st.write(f"The agent dynamically selected and executed the `{tool_name}` tool.")
                    
                    # Extract the latest message
                    if "messages" in state_update:
                        final_message = state_update["messages"][-1].content
            
            # Close the status box when done
            status.update(label="Response generated!", state="complete", expanded=False)
            
            st.markdown(final_message)
            st.session_state.messages.append({"role": "assistant", "content": final_message})
            
        except Exception as e:
            error_msg = f"**Error executing graph:** {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})