"""
Web Crawler and Content Processor for Mammo.Chat

This module provides functionality for crawling websites, processing their content,
and storing the processed data in a structured format. It includes capabilities for:

1. Sitemap Crawling:
   - Parse XML sitemaps and sitemap indices
   - Extract URLs from sitemaps following the sitemaps.org protocol

2. Content Processing:
   - Split content into manageable chunks
   - Generate embeddings using OpenAI's API
   - Extract titles and summaries using GPT models
   - Process documents in parallel with concurrency control

3. Data Storage:
   - Store processed content in Supabase
   - Handle metadata and embeddings
   - Support for different content sources

Requirements:
    - OpenAI API key for embeddings and GPT processing
    - Supabase credentials for data storage
    - Environment variables in .env file

Usage:
    Run as a script to process the Komen.org sitemap:
    ```
    python crawl_sitemap.py
    ```
    Or import functions for custom usage:
    ```
    from crawl_sitemap import process_and_store_document
    await process_and_store_document(url, content)
    ```
"""

# Standard library imports
import os
import sys
import json
import asyncio
import logging
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse

# Third-party imports
import requests
from xml.etree import ElementTree
from dotenv import load_dotenv
from openai import AsyncOpenAI
from supabase import create_client, Client
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    filename='crawl_komen_org.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

def fetch_sitemap(url: str) -> List[str]:
    """
    Fetch and parse a single XML sitemap, extracting all URLs.
    
    Args:
        url (str): The URL of the sitemap to fetch and parse
        
    Returns:
        List[str]: A list of URLs found in the sitemap
        
    Note:
        Uses the sitemaps.org XML namespace for parsing
        Prints an error message if the sitemap cannot be fetched
    """
    response = requests.get(url)
    if response.status_code == 200:
        root = ElementTree.fromstring(response.content)
        namespaces = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        return [elem.text for elem in root.findall("ns:url/ns:loc", namespaces)]
    else:
        print(f"Failed to fetch {url}. Status code: {response.status_code}")
        return []

def get_all_urls(sitemap_index_url: str = "https://www.komen.org/sitemap_index.xml") -> List[str]:
    """
    Fetch and parse a sitemap index file and all its referenced sitemaps.
    
    Args:
        sitemap_index_url (str): The URL of the sitemap index file. 
                                Defaults to Komen.org's sitemap index.
        
    Returns:
        List[str]: A list of all URLs found across all sitemaps
        
    Note:
        Processes the main sitemap index and recursively fetches all referenced sitemaps
        Prints progress messages during crawling
        Handles HTTP errors gracefully
    """
    response = requests.get(sitemap_index_url)
    if response.status_code == 200:
        root = ElementTree.fromstring(response.content)
        namespaces = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        
        # Extract individual sitemap URLs
        sitemap_urls = [elem.text for elem in root.findall("ns:sitemap/ns:loc", namespaces)]
        
        # Fetch all URLs from each sitemap
        all_urls = []
        for sitemap in sitemap_urls:
            print(f"Fetching URLs from: {sitemap}")
            all_urls.extend(fetch_sitemap(sitemap))
        return all_urls
    else:
        print(f"Failed to fetch sitemap index. Status code: {response.status_code}")
        return []

@dataclass
class ProcessedChunk:
    """
    Represents a processed chunk of content with its metadata and embeddings.
    
    Attributes:
        url: Source URL of the content
        chunk_number: Sequential number of this chunk within the document
        title: Extracted or generated title for this chunk
        summary: AI-generated summary of the chunk content
        content: Raw text content of the chunk
        metadata: Additional metadata about the chunk
        embedding: Vector embedding of the content for semantic search
    """
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = max(start + 1, end)

    return chunks

async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """
    Extract title and summary from a content chunk using GPT-4.
    
    Args:
        chunk: Text content to process
        url: Source URL of the content for context
    
    Returns:
        Dict containing 'title' and 'summary' keys with extracted information
    
    Raises:
        Exception: If the API call fails or returns invalid JSON
    """
    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative."""
    
    try:
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4-turbo-preview"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}
            ],
            response_format={ "type": "json_object" }
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error getting title and summary for {url}: {str(e)}")
        return {"title": "Error processing content", "summary": "Failed to generate summary"}

async def get_embedding(text: str) -> List[float]:
    """
    Generate an embedding vector for the given text using OpenAI's API.
    
    Args:
        text: Text content to generate embedding for
    
    Returns:
        List of floating point numbers representing the embedding vector
    
    Raises:
        Exception: If the API call fails
    """
    try:
        response = await openai_client.embeddings.create(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"Error getting embedding: {str(e)}")
        raise

async def process_chunk(
    chunk: str,
    chunk_number: int,
    url: str,
    source: str = "",
    do_embedding: bool = True,
    do_title_summary: bool = True
) -> ProcessedChunk:
    """
    Process a single chunk of text, generating title, summary, and embeddings.
    
    Args:
        chunk: Text content to process
        chunk_number: Sequential number of this chunk in the document
        url: Source URL of the content
        source: Optional source identifier
        do_embedding: Whether to generate embeddings
        do_title_summary: Whether to generate title and summary
    
    Returns:
        ProcessedChunk object containing all processed data
    """
    metadata = {
        "source": source,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "chunk_size": len(chunk)
    }
    
    # Process title and summary if requested
    title = summary = ""
    if do_title_summary:
        try:
            result = await get_title_and_summary(chunk, url)
            title = result.get("title", "")
            summary = result.get("summary", "")
        except Exception as e:
            logging.error(f"Error in title/summary for {url} chunk {chunk_number}: {str(e)}")
            title = "Error processing title"
            summary = "Error processing summary"
    
    # Generate embedding if requested
    embedding = []
    if do_embedding:
        try:
            embedding = await get_embedding(chunk)
        except Exception as e:
            logging.error(f"Error in embedding for {url} chunk {chunk_number}: {str(e)}")
    
    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=title,
        summary=summary,
        content=chunk,
        metadata=metadata,
        embedding=embedding
    )

async def insert_chunk(chunk: ProcessedChunk, source: str = "") -> None:
    """
    Insert a processed chunk into Supabase database.
    
    Args:
        chunk: ProcessedChunk object to store
        source: Optional source identifier
    
    Raises:
        Exception: If the database insertion fails
    """
    try:
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding if chunk.embedding else None,
            "source": source
        }
        
        response = await supabase.table("chunks").insert(data).execute()
        if hasattr(response, 'error') and response.error:
            raise Exception(f"Supabase error: {response.error}")
            
    except Exception as e:
        logging.error(f"Error inserting chunk {chunk.chunk_number} from {chunk.url}: {str(e)}")
        raise

async def process_and_store_document(
    url: str,
    markdown: str,
    source: str = "",
    do_embedding: bool = True,
    do_title_summary: bool = True
) -> None:
    """
    Process a document and store its chunks in parallel.
    
    Args:
        url: Source URL of the document
        markdown: Document content in markdown format
        source: Optional source identifier
        do_embedding: Whether to generate embeddings
        do_title_summary: Whether to generate title and summary
    """
    chunks = chunk_text(markdown)
    tasks = []
    
    for i, chunk in enumerate(chunks):
        task = process_chunk(
            chunk=chunk,
            chunk_number=i,
            url=url,
            source=source,
            do_embedding=do_embedding,
            do_title_summary=do_title_summary
        )
        tasks.append(task)
    
    processed_chunks = await asyncio.gather(*tasks)
    
    for chunk in processed_chunks:
        await insert_chunk(chunk, source)
        logging.info(f"Stored chunk {chunk.chunk_number} from {url}")

async def crawl_parallel(
    urls: List[str],
    max_concurrent: int = 5,
    source: str = "",
    do_embedding: bool = True,
    do_title_summary: bool = True
) -> None:
    """
    Crawl multiple URLs in parallel with a concurrency limit.
    
    Args:
        urls: List of URLs to crawl
        max_concurrent: Maximum number of concurrent crawls
        source: Optional source identifier
        do_embedding: Whether to generate embeddings
        do_title_summary: Whether to generate title and summary
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    browser_config = BrowserConfig(
        headless=True,
        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    )
    crawler = AsyncWebCrawler(browser_config)
    
    async def process_url(url: str):
        async with semaphore:
            try:
                config = CrawlerRunConfig(cache_mode=CacheMode.ALLOW_CACHE)
                result = await crawler.get_markdown(url, config)
                if result.success:
                    await process_and_store_document(
                        url=url,
                        markdown=result.markdown,
                        source=source,
                        do_embedding=do_embedding,
                        do_title_summary=do_title_summary
                    )
                    logging.info(f"Successfully processed {url}")
                else:
                    logging.error(f"Failed to crawl {url}: {result.error}")
            except Exception as e:
                logging.error(f"Error processing {url}: {str(e)}")
    
    tasks = [process_url(url) for url in urls]
    await asyncio.gather(*tasks)
    await crawler.close()

async def main() -> None:
    """
    Main entry point for the crawler.
    Crawls the Komen.org sitemap and processes all found URLs.
    """
    try:
        # Get URLs from sitemap
        urls = get_all_urls()
        if not urls:
            logging.error("No URLs found in sitemap")
            return
        
        logging.info(f"Found {len(urls)} URLs in sitemap")
        
        # Process URLs in parallel
        await crawl_parallel(
            urls=urls,
            max_concurrent=5,
            source="komen_org",
            do_embedding=True,
            do_title_summary=True
        )
        
        logging.info("Crawling completed successfully")
        
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
