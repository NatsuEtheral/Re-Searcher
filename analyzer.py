import json
import pypdf
from typing import Optional, List, Dict
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from arxiv_utils import search_arxiv

def extract_pdf_text(pdf_path: str, max_pages: int = 5) -> str:
    """
    Extract text content from the first few pages of a PDF file.
    """
    try:
        reader = pypdf.PdfReader(pdf_path)
        pages_text = []
        num_pages = len(reader.pages)
        for i in range(min(max_pages, num_pages)):
            text = reader.pages[i].extract_text()
            if text:
                pages_text.append(text)
        return "\n\n".join(pages_text)
    except Exception as e:
        raise ValueError(f"Failed to read PDF file: {e}")

def extract_keywords_and_title(text: str, groq_api_key: str) -> Dict:
    """
    Extract paper title, abstract, and search keywords using Qwen on Groq.
    """
    llm = ChatGroq(
        api_key=groq_api_key,
        model="llama-3.3-70b-versatile",
        temperature=0.0
    )
    
    system_prompt = """You are a research assistant. Extract the title, abstract, and 3-5 core search keywords from the provided paper draft text.
Your response MUST be a single JSON object. Do not write any conversational text before or after the JSON.
Format the JSON exactly like this:
{
    "title": "Title of the paper",
    "abstract": "Abstract or brief summary of the paper's goal",
    "keywords": ["keyword 1", "keyword 2", "keyword 3"]
}
"""
    
    # Send a truncated version of the draft text to avoid overloading the extraction model
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Paper draft text (first few pages):\n\n{text[:8000]}")
    ]
    
    response = llm.invoke(messages)
    content = response.content.strip()
    
    # Strip <think> tags if reasoning is output
    if "</think>" in content:
        content = content.split("</think>")[-1].strip()
        
    # Strip markdown block ticks if LLM includes them
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()
    
    try:
        return json.loads(content)
    except Exception as e:
        print(f"[DEBUG] JSON parsing failed for title/keyword extraction: {e}")
        print(f"[DEBUG] Raw content was:\n{content}\n")
        # Fallback extraction if JSON parsing fails
        return {
            "title": "Uploaded Draft",
            "abstract": "Failed to automatically parse abstract.",
            "keywords": ["deep learning", "artificial intelligence"]
        }

def run_peer_review(
    draft_text: str,
    title: str,
    abstract: str,
    baseline_papers: List[Dict],
    groq_api_key: str,
    model_name: str = "openai/gpt-oss-120b"
) -> Dict:
    """
    Run an academic peer-review evaluation of the draft text against ArXiv baseline papers.
    Returns a structured dictionary with quality scores and reviews.
    """
    llm = ChatGroq(
        api_key=groq_api_key,
        model=model_name,
        temperature=0.2
    )
    
    # Format baseline papers details
    baselines_str = ""
    for idx, p in enumerate(baseline_papers[:5]):
        baselines_str += f"\n[{idx+1}] Title: {p['title']}\nAuthors: {', '.join(p['authors'])}\nSummary: {p['summary']}\n---\n"
        
    system_prompt = """You are a senior academic reviewer for major AI and ML conferences (e.g., NeurIPS, ICML, CVPR).
Your task is to review the uploaded paper draft against the provided peer-reviewed ArXiv baseline papers and generate a comprehensive review report.

You MUST score the paper out of 100 based on the following rubric:
1. Novelty & Contribution (25 points): Evaluate the originality of the idea and its contribution compared to the baselines.
2. Methodology Rigor (25 points): Evaluate the soundness of the proposed algorithms, theoretical proofs, or experimental setups.
3. Literature & Context Alignment (25 points): Evaluate how well the paper positions itself and cites relevant baseline works.
4. Clarity & Writing Quality (25 points): Evaluate the writing quality, document structure, and ease of understanding.

Your response MUST be a single JSON object. Do not write any conversational text before or after the JSON.
Format the JSON exactly like this:
{
    "overall_score": 78,
    "novelty_score": 20,
    "novelty_feedback": "Detailed paragraph evaluating novelty...",
    "methodology_score": 18,
    "methodology_feedback": "Detailed paragraph evaluating method rigor...",
    "literature_score": 22,
    "literature_feedback": "Detailed paragraph evaluating context against peer baselines...",
    "clarity_score": 18,
    "clarity_feedback": "Detailed paragraph evaluating writing quality...",
    "strengths": [
        "First major strength of the paper",
        "Second major strength of the paper"
    ],
    "weaknesses": [
        "First weakness or gap identified",
        "Second weakness or gap identified"
    ],
    "recommendations": [
        "Actionable recommendation 1 for improving publication chances",
        "Actionable recommendation 2 for improving publication chances"
    ]
}
"""
    
    user_input = f"""Draft Paper Title: {title}
Abstract: {abstract}

Draft Paper Sample Content (First few pages):
{draft_text[:12000]}

Related Peer-Reviewed Baseline Papers:
{baselines_str}
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ]
    
    response = llm.invoke(messages)
    content = response.content.strip()
    
    # Strip <think> tags if reasoning is output
    if "</think>" in content:
        content = content.split("</think>")[-1].strip()
        
    # Strip markdown block ticks if LLM includes them
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()
    
    try:
        return json.loads(content)
    except Exception as e:
        # Fallback basic review if LLM output fails to parse
        raise ValueError(f"Failed to generate structured review JSON: {e}\nRaw output: {content}")

def analyze_paper_draft(pdf_path: str, groq_api_key: str, model_name: str = "llama-3.3-70b-versatile") -> Dict:
    """
    Orchestrate the complete paper analysis workflow:
    1. Extract PDF text.
    2. Extract keywords.
    3. Search ArXiv for related baseline papers.
    4. Run peer-review grading and evaluation.
    """
    # 1. Parse draft text
    text = extract_pdf_text(pdf_path, max_pages=5)
    if not text.strip():
        raise ValueError("The uploaded PDF has no extractable text. Please ensure it is not a scanned image PDF.")
        
    # 2. Extract metadata & keywords
    metadata = extract_keywords_and_title(text, groq_api_key)
    title = metadata.get("title", "Uploaded Draft")
    abstract = metadata.get("abstract", "")
    keywords = metadata.get("keywords", [])
    
    # 3. Retrieve related papers
    baseline_papers = []
    if keywords:
        # Combine keywords with OR for broader search
        search_query = " OR ".join([f'"{kw}"' for kw in keywords])
        baseline_papers = search_arxiv(search_query, max_results=6)
        
    # Fallback search if combined fails
    if not baseline_papers and keywords:
        baseline_papers = search_arxiv(keywords[0], max_results=6)
        
    # 4. Generate Review
    review_report = run_peer_review(
        draft_text=text,
        title=title,
        abstract=abstract,
        baseline_papers=baseline_papers,
        groq_api_key=groq_api_key,
        model_name=model_name
    )
    
    # Add metadata to output
    review_report["title"] = title
    review_report["abstract"] = abstract
    review_report["keywords"] = keywords
    review_report["baseline_papers"] = baseline_papers
    
    return review_report
