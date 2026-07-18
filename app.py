import streamlit as st
import shutil
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

import importlib
import config
import vector_db
import agent

# Force reload helper modules during development so changes are picked up on refresh
importlib.reload(config)
importlib.reload(vector_db)
importlib.reload(agent)

from agent import agent_graph

# ----------------------------------------------------
# 1. Page Configuration & Custom CSS
# ----------------------------------------------------
st.set_page_config(
    page_title="Re-Searcher | ArXiv Chatbot",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sleek CSS styling for rich aesthetics
st.markdown("""
<style>
    /* Main container adjustments */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header styling */
    .title-container {
        display: flex;
        align-items: center;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid #2e303d;
        padding-bottom: 0.8rem;
    }
    .title-icon {
        font-size: 2.5rem;
        margin-right: 1rem;
    }
    .title-text {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #60a5fa, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Card design for paper list */
    .paper-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        transition: transform 0.2s, border-color 0.2s;
    }
    .paper-card:hover {
        border-color: #3b82f6;
        transform: translateY(-2px);
    }
    .paper-title {
        color: #f8fafc;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    .paper-authors {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-bottom: 0.4rem;
    }
    .paper-meta {
        display: flex;
        gap: 1rem;
        font-size: 0.8rem;
        color: #64748b;
        margin-bottom: 0.6rem;
    }
    .paper-summary {
        color: #cbd5e1;
        font-size: 0.9rem;
        line-height: 1.4;
    }
    
    /* Custom info message */
    .custom-info {
        background-color: #0f172a;
        border-left: 4px solid #3b82f6;
        color: #94a3b8;
        padding: 0.8rem;
        border-radius: 4px;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. Session State Initialization
# ----------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "active_paper_id" not in st.session_state:
    st.session_state.active_paper_id = None

# Helper to clear local database
def reset_vector_db():
    try:
        # Clear vector database directory
        if config.PERSIST_DIR.exists():
            shutil.rmtree(config.PERSIST_DIR)
        config.PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        
        # Clear downloads directory
        if config.DOWNLOAD_DIR.exists():
            shutil.rmtree(config.DOWNLOAD_DIR)
        config.DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
        
        st.session_state.active_paper_id = None
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.success("Vector Database and Downloads successfully cleared!")
    except Exception as e:
        st.error(f"Error resetting database: {e}")

# ----------------------------------------------------
# 3. Sidebar UI (Settings & Indexed Library)
# ----------------------------------------------------
with st.sidebar:
    st.markdown("## 🔬 Re-Searcher")
    st.markdown("### Settings")
    
    # API Keys Configuration
    groq_key = st.text_input(
        "Groq API Key",
        value=config.GROQ_API_KEY,
        type="password",
        help="Input your Groq API key here. Defaults to environment variable if set."
    )
    
    google_key = st.text_input(
        "Google API Key (Optional)",
        value=config.GOOGLE_API_KEY,
        type="password",
        help="Optional. Input Google AI Studio key to use Gemini Embeddings."
    )
    
    # Model Selection (Global Var Configuration)
    model_options = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768"
    ]
    groq_model = st.selectbox(
        "Groq Model",
        options=model_options,
        index=0,
        help="Choose the model to run the LangGraph agent."
    )
    
    # Embeddings Provider Selection
    if google_key.strip():
        emb_provider = st.selectbox("Embedding Provider", options=["Gemini", "Local HuggingFace"], index=0)
    else:
        st.info("Provide a Google API key to enable Gemini embeddings.")
        emb_provider = st.selectbox("Embedding Provider", options=["Local HuggingFace"], index=0, disabled=True)
        
    embedding_provider_val = "gemini" if emb_provider == "Gemini" else "huggingface"
    
    # Store settings in session state so tools can access them
    st.session_state.google_api_key = google_key.strip() if google_key.strip() else None
    st.session_state.embedding_provider = embedding_provider_val
    
    st.markdown("---")
    st.markdown("### 📚 Indexed Papers Library")
    
    # Fetch indexed papers from manifest
    indexed_papers = vector_db.get_indexed_papers()
    
    if indexed_papers:
        # Create a list for selection
        paper_options = {"None (Search across all or use general chat)": None}
        for paper in indexed_papers:
            # Format label: [arxiv_id] Title
            label = f"[{paper['arxiv_id']}] {paper['title'][:40]}..."
            paper_options[label] = paper["arxiv_id"]
            
        selected_label = st.radio(
            "Select Active Paper Context:",
            options=list(paper_options.keys()),
            index=0
        )
        st.session_state.active_paper_id = paper_options[selected_label]
        
        # Display metadata of active paper
        if st.session_state.active_paper_id:
            active_metadata = next(p for p in indexed_papers if p["arxiv_id"] == st.session_state.active_paper_id)
            st.markdown(f"**Title**: {active_metadata['title']}")
            st.markdown(f"**Authors**: {', '.join(active_metadata['authors'][:3])}...")
            st.markdown(f"[PDF Link]({active_metadata['pdf_url']})")
    else:
        st.write("No papers indexed yet. Search for a paper and ask to index it.")
        st.session_state.active_paper_id = None
        
    st.markdown("---")
    if st.button("Reset Database", type="secondary", help="Clear all stored documents and start fresh"):
        reset_vector_db()
        st.rerun()

# ----------------------------------------------------
# 4. Main Panel UI
# ----------------------------------------------------
st.markdown("""
<div class="title-container">
    <div class="title-icon">🔬</div>
    <div class="title-text">Re-Searcher</div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="custom-info">Welcome to Re-Searcher, your ArXiv AI research companion! '
    'Ask me to search for papers, index specific PDFs, and perform deep Q&A retrieval.</div>',
    unsafe_allow_html=True
)

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input & Orchestration
if prompt := st.chat_input("Ask a question, e.g., 'Search for papers on direct preference optimization'"):
    # Check Groq Key
    if not groq_key.strip():
        st.error("Please provide a Groq API Key in the sidebar settings to proceed.")
        st.stop()
        
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Append message to LangGraph history
    st.session_state.chat_history.append(HumanMessage(content=prompt))
    
    # Prepare inputs for the Graph
    inputs = {"messages": st.session_state.chat_history}
    
    # If active paper context is set, append system notification to inform the agent
    if st.session_state.active_paper_id:
        inputs["messages"] = list(st.session_state.chat_history)
        inputs["messages"].insert(-1, SystemMessage(
            content=f"System Context: The user has selected the paper with ArXiv ID '{st.session_state.active_paper_id}' "
                    f"as the active context. Any questions regarding 'this paper', 'it', or 'the findings' should query this paper "
                    f"using ask_about_paper."
        ))
        
    # Compile graph execution configuration
    exec_config = {
        "configurable": {
            "groq_api_key": groq_key.strip(),
            "google_api_key": google_key.strip() if google_key.strip() else None,
            "embedding_provider": embedding_provider_val,
            "groq_model": groq_model
        }
    }
    
    # Display assistant response placeholder
    with st.chat_message("assistant"):
        response_text = ""
        # Create expandable status container for tool execution trace
        with st.status("🔬 Thinking and orchestrating...", expanded=True) as status:
            try:
                for update in agent_graph.stream(inputs, exec_config, stream_mode="updates"):
                    for node, value in update.items():
                        if node == "tools":
                            for msg in value.get("messages", []):
                                status.write(f"🔧 **Tool `{msg.name}` completed execution.**")
                                # Provide clean expandable sections for the tool outputs
                                with st.expander("Show tool details", expanded=False):
                                    status.code(msg.content[:1000] + ("..." if len(msg.content) > 1000 else ""))
                        elif node in ("chatbot", "router", "reasoner"):
                            last_msg = value.get("messages", [])[-1]
                            if last_msg.content:
                                response_text = last_msg.content
                            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                                for tc in last_msg.tool_calls:
                                    status.write(f"🤖 **Agent decided to invoke tool**: `{tc['name']}`")
                
                status.update(label="Response generated!", state="complete", expanded=False)
            except Exception as e:
                status.update(label="Error occurred", state="error", expanded=True)
                st.error(f"Execution Error: {e}")
                
        # Render the final response if generated
        if response_text:
            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            st.session_state.chat_history.append(AIMessage(content=response_text))
            
            # Simple UI reload trigger to refresh sidebar indexed list in case a new paper was indexed
            if "Successfully downloaded and indexed paper" in response_text or "already indexed" in response_text:
                st.rerun()
        else:
            st.warning("The agent completed run but did not return a final response content.")
