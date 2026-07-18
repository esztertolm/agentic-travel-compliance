import logging
from typing import TypedDict, Annotated
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from configuration.config import AppConfig
from tools import convert_currency
from rag_subgraph import policy_search_tool 

logger = logging.getLogger(__name__)


class AssistantState(TypedDict):
    # 'add_messages' ensures new messages are appended to the list instead of overwriting
    messages: Annotated[list, add_messages]
    categories: list[str]
    is_approved: bool
    audit_status: str

class InputCategories(BaseModel):
    categories: list[str] = Field(
        description="A list of categories that apply to the user's request. Select one or more from: 'Travel Policy', 'Finance', 'General'."
    )

# Define available tools
TOOLS = [policy_search_tool, convert_currency]

def categorize_input_node(state: AssistantState) -> AssistantState:
    """Node 1: LLM-based multi-label categorization logic for semantic routing and metadata."""
    logger.info("[Main Graph] Node 1: Categorizing user input using LLM...")
    last_message = state["messages"][-1].content
    
    llm = ChatOllama(model=AppConfig.LLM_MODEL_NAME, temperature=0.0, base_url=AppConfig.OLLAMA_BASE_URL)
    parser = PydanticOutputParser(pydantic_object=InputCategories)
    
    prompt = PromptTemplate(
        template="""Analyze the following user input and determine ALL appropriate categories based on the context.\n
                {format_instructions}\n\n
                User Input: {input}\n\nIMPORTANT: Return ONLY a valid JSON object!""",
        input_variables=["input"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    try:
        raw_response = llm.invoke(prompt.format(input=last_message))
        content = raw_response.content.replace("```json", "").replace("```", "").strip()
        
        parsed_result = parser.parse(content)
        categories = parsed_result.categories
        
        if not categories:
            categories = ["General"]
            
    except Exception as e:
        logger.warning("LLM categorization failed, defaulting to ['General']: %s", str(e))
        categories = ["General"]
        
    logger.info("Assigned semantic categories: %s", categories)
    return {"categories": categories}

def agent_reasoning_node(state: AssistantState) -> AssistantState:
    """Node 2: The Core LLM logic that decides to use tools or answer directly."""
    logger.info("[Main Graph] Node 2: Agent reasoning and tool binding...")
    llm = ChatOllama(model=AppConfig.LLM_MODEL_NAME, temperature=0.0, base_url=AppConfig.OLLAMA_BASE_URL)
    
    # Bind tools directly to the LLM
    llm_with_tools = llm.bind_tools(TOOLS)
    
    # The LLM reads the entire conversation history and decides the next move
    response = llm_with_tools.invoke(state["messages"])
    
    # Return the AIMessage (which may contain tool_calls or the final text)
    return {"messages": [response]}

def quality_check_node(state: AssistantState) -> AssistantState:
    """Node 4: Evaluates the AI's final answer."""
    logger.info("[Main Graph] Node 4: Running quality assurance check...")
    llm = ChatOllama(model=AppConfig.LLM_MODEL_NAME, temperature=0.0, base_url=AppConfig.OLLAMA_BASE_URL)
    
    # The last message is the AI's final synthesized response
    draft_response = state["messages"][-1].content
    
    prompt = f"""Evaluate if this response is helpful and safe:
            Response: {draft_response}
            Reply EXACTLY with 'APPROVED' or 'REJECTED'."""
    
    eval_response = llm.invoke([HumanMessage(content=prompt)]).content.strip().upper()
    is_approved = "APPROVED" in eval_response
    
    return {"is_approved": is_approved}

def audit_log_node(state: AssistantState) -> AssistantState:
    """Node 5: Finalizes the transaction and saves metadata."""
    logger.info("[Main Graph] Node 5: Saving Audit Log...")
    logger.info("Transaction Category: %s | Compliance Approved: %s", state.get("category"), state.get("is_approved"))
    return {"audit_status": "Completed"}

def route_after_reasoning(state: AssistantState) -> str:
    """Determines if the LLM requested a tool call or provided a final answer."""
    last_message = state["messages"][-1]
    
    # If the LLM generated tool_calls, route to the ToolNode
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        logger.info("Routing to tools: %s", [t["name"] for t in last_message.tool_calls])
        return "action_node"
    
    # Otherwise, the LLM gave a final answer, move to quality check
    return "quality_check_node"


def build_main_graph() -> StateGraph:
    graph = StateGraph(AssistantState)
    
    # Add Nodes
    graph.add_node("categorize_input_node", categorize_input_node)
    graph.add_node("agent_reasoning_node", agent_reasoning_node)
    graph.add_node("action_node", ToolNode(TOOLS))
    graph.add_node("quality_check_node", quality_check_node)
    graph.add_node("audit_log_node", audit_log_node)
    
    # Edges
    graph.add_edge(START, "categorize_input_node")
    graph.add_edge("categorize_input_node", "agent_reasoning_node")
    
    # Conditional edge from the LLM
    graph.add_conditional_edges(
        "agent_reasoning_node",
        route_after_reasoning,
        {
            "action_node": "action_node",
            "quality_check_node": "quality_check_node"
        }
    )
    
    # The ToolNode always returns back to the LLM so it can read the tool's output!
    graph.add_edge("action_node", "agent_reasoning_node")
    
    graph.add_edge("quality_check_node", "audit_log_node")
    graph.add_edge("audit_log_node", END)
    
    memory = MemorySaver()
    return graph.compile(checkpointer=memory, name="travel_compliance_agent")

def initialize_agent():
    return build_main_graph()

if __name__ == "__main__":
    from utils.logging import setup_logger
    setup_logger()
    logger.info("=== Starting Modern LangGraph ToolNode Agent ===")
    
    app = initialize_agent()
    config = {"configurable": {"thread_id": "test_modern_1"}}
    
    initial_input = "What is the maximum allowance for a hotel stay per night, and how much is that in EUR if I travel to Paris?"
    
    # We now pass a HumanMessage object into the messages list
    test_state = {"messages": [HumanMessage(content=initial_input)]}
    
    response_state = app.invoke(test_state, config=config)
    
    logger.info("--- FINAL AI RESPONSE ---")
    logger.info(response_state["messages"][-1].content)