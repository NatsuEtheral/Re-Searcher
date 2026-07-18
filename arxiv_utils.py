import arxiv
import re

def search_arxiv(query: str, max_results: int = 10) -> list[dict]:
    """
    Search ArXiv for research papers based on a query.
    Returns a list of structured dictionaries.
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    results = []
    try:
        for result in client.results(search):
            # Extract clean ID from full entry_id URI (e.g. http://arxiv.org/abs/2305.18290v1 -> 2305.18290v1)
            arxiv_id = result.entry_id.split("/abs/")[-1]
            # Strip version suffix (e.g., v1, v2) if we want the base ID for downloading
            base_id = re.sub(r'v\d+$', '', arxiv_id)
            
            results.append({
                "entry_id": result.entry_id,
                "arxiv_id": arxiv_id,
                "base_id": base_id,
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "summary": result.summary.replace("\n", " ").strip(),
                "published": result.published.strftime("%Y-%m-%d"),
                "pdf_url": result.pdf_url,
                "primary_category": result.primary_category
            })
    except Exception as e:
        print(f"Error searching ArXiv: {e}")
        
    return results
