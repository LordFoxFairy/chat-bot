from langchain_core.tools import tool

@tool
def search_web(query: str) -> str:
    """Search the web for information about the given query."""
    # This is a mock implementation
    return f"Simulated search results for: {query}\n1. Link 1\n2. Link 2\n3. Link 3"
