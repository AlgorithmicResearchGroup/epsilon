import os

import trafilatura
from tavily import TavilyClient


web_tool_definitions = [
    {
        "name": "web_search",
        "description": "Search the web using Tavily. Returns a list of results with title, URL, and snippet.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return. Default: 5.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_url",
        "description": "Fetch a URL and extract its text content. Returns clean readable text, not raw HTML. Good for reading articles, docs, references.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch.",
                },
            },
            "required": ["url"],
        },
    },
]


def web_search(arguments, work_dir=None):
    query = arguments["query"]
    num_results = arguments.get("num_results", 5)

    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    response = client.search(query=query, max_results=num_results)

    results = response["results"]
    formatted_lines = []
    for i, r in enumerate(results, 1):
        formatted_lines.append(f"{i}. {r['title']}\n   URL: {r['url']}\n   {r['content'][:300]}")

    output = "\n\n".join(formatted_lines) if formatted_lines else "No results found."

    return {
        "tool": "web_search",
        "status": "success",
        "stdout": output,
        "stderr": "",
    }


def fetch_url(arguments, work_dir=None):
    url = arguments["url"]

    downloaded = trafilatura.fetch_url(url)
    text = trafilatura.extract(downloaded)

    if not text:
        return {
            "tool": "fetch_url",
            "status": "error",
            "stdout": "",
            "stderr": f"Could not extract text from {url}",
        }

    return {
        "tool": "fetch_url",
        "status": "success",
        "stdout": text,
        "stderr": "",
    }
