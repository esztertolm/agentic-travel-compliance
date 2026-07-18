import logging
from typing import TypedDict
from langgraph.graph import END, StateGraph

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from configuration.config import AppConfig
from database import get_retriever

logger = logging.getLogger(__name__)

class RAGState(TypedDict):
    query: str
    documents: str
    rag_answer: str

def retrieve_docs(state: RAGState) -> RAGState:
    logger.info("[RAG Subgraph] Retrieving documents for query: %s", state["query"])
    retriever = get_retriever()
    docs = retriever.invoke(state["query"])
    
    doc_text = "\n\n".join([
        f"Source: {d.metadata.get('source_file')}, Page {d.metadata.get('page')}\n{d.page_content}" 
        for d in docs
    ])
    return {"query": state["query"], "documents": doc_text, "rag_answer": ""}

def generate_rag_answer(state: RAGState) -> RAGState:
    logger.info("[RAG Subgraph] Generating answer based on retrieved documents...")
    llm = ChatOllama(model=AppConfig.LLM_MODEL_NAME, temperature=0.0, base_url=AppConfig.OLLAMA_BASE_URL)
    
    prompt = f"""Answer the question based ONLY on the following corporate travel policy documents.
                Documents: {state['documents']}
                Question: {state['query']}"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"query": state["query"], "documents": state["documents"], "rag_answer": response.content}

def build_rag_subgraph() -> StateGraph:
    """Builds the modular RAG subgraph."""
    rag_graph = StateGraph(RAGState)
    rag_graph.add_node("retrieve_docs", retrieve_docs)
    rag_graph.add_node("generate_rag_answer", generate_rag_answer)
    rag_graph.set_entry_point("retrieve_docs")
    rag_graph.add_edge("retrieve_docs", "generate_rag_answer")
    rag_graph.add_edge("generate_rag_answer", END)
    return rag_graph.compile()


RAG_SUBGRAPH_APP = build_rag_subgraph()


@tool
def policy_search_tool(query: str) -> str:
    """
    Search and retrieve information from the official corporate travel policy documents.
    ALWAYS use this tool first when the user asks about travel rules, allowances, or reimbursements.
    """
    logger.info("Executing policy_search_tool (Invoking pre-built RAG Subgraph)...")
    result = RAG_SUBGRAPH_APP.invoke({"query": query, "documents": "", "rag_answer": ""})
    return result["rag_answer"]