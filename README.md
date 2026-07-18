# 🔬 Re-Searcher: ArXiv AI Research Companion

**Re-Searcher** is a powerful AI-driven research assistant designed to query the ArXiv open-source research database, index relevant papers into a vector database, and perform context-aware, deep retrieval-augmented generation (RAG) Q&A.

Built on a robust multi-model routed agent architecture, Re-Searcher uses dedicated LLMs to route actions and generate professional-grade, cited responses.

---

## 🚀 Key Features

*   **Smart Paper Search**: Query ArXiv with keywords, subject categories, or authors, retrieving up to 10 relevant papers at a time.
*   **Vector Database Indexing**: Download and parse specific ArXiv PDF papers into a local **Qdrant** database with one click.
*   **Deep Contextual Q&A**: Ask detailed questions about any indexed paper's methodology, experiments, or findings, with precise **page number citations**.
*   **Hybrid Embeddings**: Supports both **Gemini Text Embedding** models (requires Google API Key) and local **HuggingFace Sentence Transformers** for offline vector generation.
*   **Dynamic Settings**: Swap embedding models, reasoning models, and API keys directly from the responsive Streamlit sidebar.

---

## 🧠 Routed Agent Architecture

Re-Searcher is designed to prevent function-calling failures and optimize generation quality using a **Routed Multi-Model Agent Graph** built on **LangGraph**:

```mermaid
graph TD
    Start([User Query]) --> Router[Router Node: Qwen 27B]
    Router -->|Wants Tool Call| Tools[Tools Node: search_papers / index / Q&A]
    Tools --> Reasoner[Reasoner Node: Llama 3.3 70B]
    Router -->|Direct Answer| Reasoner
    Reasoner --> End([Markdown Response])
```

1.  **Router Node (`qwen/qwen3.6-27b`)**: Qwen is highly resilient at JSON tool invocation. It receives the conversation history and selects the correct tool (or directly answers the user) without generating malformed token structures.
2.  **Tools Execution Node**: Executes Python tools (ArXiv API requests, PDF parser, and Qdrant database queries).
3.  **Reasoner Node (`llama-3.3-70b-versatile`)**: Receives the original user request and the tool outputs to synthesize the final markdown response. Because no tools are bound to Llama 70B, it is immune to function-calling crashes, allowing it to focus entirely on reasoning and high-fidelity output.

---

## 🛠️ Technology Stack

*   **Frontend**: Streamlit
*   **Orchestration**: LangGraph, LangChain
*   **Models**: Qwen 27B (routing), Llama-3.3-70B (reasoning) via Groq
*   **Vector Database**: Qdrant
*   **Embeddings**: Gemini API or HuggingFace (`all-MiniLM-L6-v2`)
*   **Environment**: Python 3.12

---

## ⚙️ Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.12 installed on your system.

### 2. Clone and Setup Environment
Clone the repository:
```bash
git clone https://github.com/NatsuEtheral/Re-Searcher.git
cd Re-Searcher
```

Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure API Keys
Create a `.env` file in the root directory:
```env
GROQ_API_KEY="your-groq-api-key"
GOOGLE_API_KEY="your-google-ai-studio-key"  # Optional: For Gemini Embeddings
```

---

## 🏃 Running the Application

Start the Streamlit application:
```bash
streamlit run app.py
```

Open your browser and navigate to the local URL (usually `http://localhost:8501`).
