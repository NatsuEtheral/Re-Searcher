from typing import TypedDict, Annotated, Sequence, Union
import operator
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

import config
from arxiv_utils import search_arxiv
from vector_db import (
    index_paper as index_paper_func,
    query_paper,
    is_paper_indexed
)

# Define State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# Define Tools
@tool
def search_papers(query: str) -> str:
    """
    Search ArXiv for research papers.
    """
    results = search_arxiv(query, max_results=10)
    if not results:
        return "No papers found matching the query."
    
    output = []
    for p in results:
        output.append(
            f"Title: {p['title']}\n"
            f"Authors: {', '.join(p['authors'])}\n"
            f"Published: {p['published']}\n"
            f"ArXiv ID: {p['arxiv_id']}\n"
            f"PDF URL: {p['pdf_url']}\n"
            f"Summary: {p['summary']}\n"
            f"---"
        )
    return "\n\n".join(output)

@tool
def index_paper(arxiv_id: Union[str, float, int]) -> str:
    """
    Download and index a specific ArXiv paper by its ID.
    """
    arxiv_id = str(arxiv_id).strip()
    if is_paper_indexed(arxiv_id):
        return f"Paper {arxiv_id} is already indexed. You can ask questions about its content now!"
        
    # We query metadata from ArXiv first
    results = search_arxiv(query=arxiv_id, max_results=1)
    if not results:
        # Fallback to search using client directly if search_arxiv fails to retrieve it
        import arxiv
        try:
            client = arxiv.Client()
            search = arxiv.Search(id_list=[arxiv_id])
            res = next(client.results(search))
            paper_metadata = {
                "arxiv_id": arxiv_id,
                "title": res.title,
                "authors": [a.name for a in res.authors],
                "summary": res.summary.replace("\n", " ").strip(),
                "published": res.published.strftime("%Y-%m-%d"),
                "pdf_url": res.pdf_url,
                "primary_category": res.primary_category
            }
        except Exception as e:
            return f"Failed to retrieve metadata for paper ID {arxiv_id}: {e}"
    else:
        paper_metadata = results[0]
        
    # Retrieve configuration from Streamlit session state dynamically
    import streamlit as st
    google_api_key = None
    provider = None
    try:
        if st.session_state:
            google_api_key = st.session_state.get("google_api_key", None)
            provider = st.session_state.get("embedding_provider", None)
    except Exception:
        pass
        
    success = index_paper_func(paper_metadata, provider=provider, google_api_key=google_api_key)
    if success:
        return f"Successfully downloaded and indexed paper '{paper_metadata['title']}' (ID: {arxiv_id}). You can now ask questions about its content using the ask_about_paper tool!"
    else:
        return f"Failed to download or index paper {arxiv_id}."

@tool
def ask_about_paper(arxiv_id: Union[str, float, int], question: str) -> str:
    """
    Query the content of an indexed paper to answer a question.
    """
    arxiv_id = str(arxiv_id).strip()
    if not is_paper_indexed(arxiv_id):
        return f"Paper {arxiv_id} is not indexed yet. Please call the 'index_paper' tool on it first."
        
    import streamlit as st
    google_api_key = None
    provider = None
    try:
        if st.session_state:
            google_api_key = st.session_state.get("google_api_key", None)
            provider = st.session_state.get("embedding_provider", None)
    except Exception:
        pass
        
    chunks = query_paper(arxiv_id, question, k=4, provider=provider, google_api_key=google_api_key)
    if not chunks:
        return "No relevant sections found in the paper for this question."
        
    context_parts = []
    for i, chunk in enumerate(chunks):
        page = chunk.metadata.get("page", "unknown")
        context_parts.append(f"[Chunk {i+1} | Page {page}]:\n{chunk.page_content}")
        
    return "\n\n".join(context_parts)

# Define the Tool List
tools = [search_papers, index_paper, ask_about_paper]
tool_node = ToolNode(tools)

# 1. Router Node: Uses Qwen 27b to decide tools
def call_router(state: AgentState, config: RunnableConfig):
    # Extract config parameters from execution context
    configurable = config.get("configurable", {})
    groq_api_key = configurable.get("groq_api_key", "") or config.GROQ_API_KEY
    
    if not groq_api_key:
        raise ValueError("Groq API Key is missing. Please set it in your environment or provide it in the UI.")
        
    router_llm = ChatGroq(
        api_key=groq_api_key,
        model="qwen/qwen3.6-27b",
        temperature=0.0
    )
    router_with_tools = router_llm.bind_tools(tools)
    
    SYSTEM_PROMPT = """You are a router agent. Your sole job is to look at the user prompt and decide if you need to use one of the available tools.
If you need to use a tool, invoke it with the correct arguments.
If you do not need any tool to answer the user request, do not call any tool and respond with text.
Available tools:
- search_papers: to search for papers on ArXiv.
- index_paper: to download and index a specific paper.
- ask_about_paper: to ask questions about the contents of an indexed paper.

Crucial Guidelines:
1. If the user asks a question about a paper's content, you must query its content using the `ask_about_paper` tool.
2. If `ask_about_paper` returns that the paper is not indexed, you must call the `index_paper` tool to download and index it first.
3. Once `index_paper` returns success, you must call the `ask_about_paper` tool to query the paper's contents and retrieve the text before handing off to the reasoner.
"""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
    
    print("\n================ [DEBUG] ROUTER NODE START ================")
    print("Node: router")
    print("Model: qwen/qwen3.6-27b")
    print(f"Number of messages in history: {len(state['messages'])}")
    for idx, msg in enumerate(messages):
        print(f"Message {idx}: Type={type(msg).__name__} | Content: {repr(msg.content[:120])}")
    print("===========================================================\n")
    
    try:
        response = router_with_tools.invoke(messages)
        print("\n================ [DEBUG] ROUTER NODE SUCCESS ================")
        print(f"Response Content: {repr(response.content[:200])}")
        print(f"Tool Calls: {getattr(response, 'tool_calls', [])}")
        print("=============================================================\n")
        return {"messages": [response]}
    except Exception as e:
        print("\n❌❌❌ [DEBUG] ROUTER NODE FAILED! ❌❌❌")
        print(f"Model: qwen/qwen3.6-27b")
        print(f"Exception: {e}")
        print("=========================================================\n")
        raise e

# 2. Router Routing Edge
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "reasoner"

def call_reasoner(state: AgentState, config: RunnableConfig):
    # Extract config parameters from execution context
    configurable = config.get("configurable", {})
    groq_api_key = configurable.get("groq_api_key", "") or config.GROQ_API_KEY
    groq_model = configurable.get("groq_model", "") or config.GROQ_MODEL
    
    if not groq_api_key:
        raise ValueError("Groq API Key is missing. Please set it in your environment or provide it in the UI.")
        
    reasoner_llm = ChatGroq(
        api_key=groq_api_key,
        model=groq_model,
        temperature=0.2
    )
    
    SYSTEM_PROMPT = """You are Re-Searcher, a powerful AI research assistant. 
You help users find, summarize, and understand research papers from ArXiv.

You are in synthesis mode. Your sole job is to formulate a clear, helpful response to the user using the search results or paper context already retrieved in the conversation history.

Guidelines:
- When presenting search results, you MUST list all the papers retrieved in a numbered list, showing their Title, Authors, Published Date, ID, PDF link, and a brief 1-2 sentence summary of their abstract. Do not summarize them into a single high-level paragraph.
- When answering from paper contents (context returned by the tool), mention the relevant page number citations (e.g. Page 3) based on the context.
- Keep your tone professional, scientific, and helpful.
- Do not attempt to invoke any more tool calls. Answer the user directly.
"""
    
    # Sanitize message history to remove native tool_calls and ToolMessage structures
    # so that the API gateway doesn't complain about schema/tool mismatches on non-tool-calling models.
    sanitized_messages = []
    for msg in state["messages"]:
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            # Strip tool_calls from AIMessage and convert to simple text-based AIMessage
            calls_desc = ", ".join([f"{tc['name']}({tc['args']})" for tc in msg.tool_calls])
            sanitized_messages.append(AIMessage(content=f"[Assistant invoked tool(s): {calls_desc}]"))
        elif isinstance(msg, ToolMessage):
            # Convert ToolMessage output to a clean SystemMessage
            sanitized_messages.append(SystemMessage(
                content=f"[System Info: Tool '{msg.name}' completed execution and returned results:\n{msg.content}]"
            ))
        else:
            sanitized_messages.append(msg)
            
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + sanitized_messages
    
    print("\n================ [DEBUG] REASONER NODE START ================")
    print("Node: reasoner")
    print(f"Model: {groq_model}")
    print(f"Number of messages in history: {len(sanitized_messages)}")
    for idx, msg in enumerate(messages):
        print(f"Message {idx}: Type={type(msg).__name__} | Content: {repr(msg.content[:120])}")
    print("=============================================================\n")
    
    try:
        response = reasoner_llm.invoke(messages)
        print("\n================ [DEBUG] REASONER NODE SUCCESS ================")
        print(f"Response Content: {repr(response.content[:200])}")
        print("===============================================================\n")
        return {"messages": [response]}
    except Exception as e:
        print("\n❌❌❌ [DEBUG] REASONER NODE FAILED! ❌❌❌")
        print(f"Model: {groq_model}")
        print(f"Exception: {e}")
        print("===========================================================\n")
        raise e

# Build the Graph Workflow
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("router", call_router)
workflow.add_node("tools", tool_node)
workflow.add_node("reasoner", call_reasoner)

# Set Edges
workflow.add_edge(START, "router")

workflow.add_conditional_edges(
    "router",
    should_continue,
    {
        "tools": "tools",
        "reasoner": "reasoner"
    }
)

# After tools run, we route back to the router node to allow chaining multiple tool calls
workflow.add_edge("tools", "router")
workflow.add_edge("reasoner", END)

# Compile the Graph
agent_graph = workflow.compile()
