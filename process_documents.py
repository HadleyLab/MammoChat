"""Document Processing Pipeline for MammoChat.

This module provides a comprehensive pipeline for processing medical documentation
in two phases:

1. Crawling & Storage Phase:
   - Crawl trusted medical websites
   - Store raw content without AI processing
   - Support for concurrent crawling

2. AI Processing Phase:
   - Process stored content with OpenAI APIs
   - Generate embeddings for semantic search
   - Create summaries for content chunks

The pipeline is designed to be memory efficient and fault-tolerant, with
comprehensive logging and error handling throughout the process.

Typical usage:
    # Phase 1: Crawl and store raw content
    python process_documents.py crawl --source komen_org --urls https://www.komen.org/treatment

    # Phase 2: Process stored content with AI
    python process_documents.py process --batch-size 50
"""

from __future__ import annotations

import os
import sys
import json
import asyncio
import logging
import argparse
from typing import List, Dict, Any, Optional, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

# Third-party imports
import openai
from openai.types.chat import ChatCompletion
from supabase import create_client, Client
from tqdm import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

from config import config

@dataclass
class ProcessingConfig:
    """Configuration settings for the document processing pipeline.
    
    This class defines all configurable parameters for both the crawling
    and processing phases of the pipeline.
    
    Attributes:
        MAX_CONCURRENT: Maximum number of concurrent crawls
        CHUNK_SIZE: Target size for content chunks
        MIN_CHUNK_LENGTH: Minimum length for a valid chunk
        BROWSER_HEADLESS: Whether to run browser in headless mode
        BROWSER_USER_AGENT: User agent string for crawling
        CACHE_MODE: Cache mode for crawler
        BATCH_SIZE: Number of pages to process in each batch
        MAX_RETRIES: Maximum number of retry attempts
        RETRY_MIN_WAIT: Minimum wait time between retries
        RETRY_MAX_WAIT: Maximum wait time between retries
        TABLE: Database table name
        LOG_FORMAT: Format string for log messages
        LOG_LEVEL: Logging level
        LOG_FILE: Path to log file
        SYSTEM_PROMPT: Prompt for AI title/summary generation
    """
    
    # Crawling Settings
    MAX_CONCURRENT: int = 5
    CHUNK_SIZE: int = 5000
    MIN_CHUNK_LENGTH: int = 100
    BROWSER_HEADLESS: bool = True
    BROWSER_USER_AGENT: str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    CACHE_MODE: CacheMode = CacheMode.ENABLED
    
    # Processing Settings
    BATCH_SIZE: int = 50
    MAX_RETRIES: int = 3
    RETRY_MIN_WAIT: int = 4
    RETRY_MAX_WAIT: int = 10
    
    # Database Settings
    TABLE: str = "site_pages"
    
    # Logging Settings
    LOG_FORMAT: str = "%(asctime)s - %(levelname)s - %(message)s"
    LOG_LEVEL: int = logging.INFO
    LOG_FILE: str = "document_pipeline.log"
    
    # AI Processing Settings
    SYSTEM_PROMPT: str = """You are an AI that extracts titles and summaries from documentation chunks.
        Return a JSON object with 'title' and 'summary' keys.
        For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
        For the summary: Create a concise summary of the main points in this chunk.
        Keep both title and summary concise but informative."""

# Configure logging
logging.basicConfig(
    level=ProcessingConfig.LOG_LEVEL,
    format=ProcessingConfig.LOG_FORMAT,
    handlers=[
        logging.FileHandler(ProcessingConfig.LOG_FILE),
        logging.StreamHandler()
    ]
)

class ProcessingPhase(Enum):
    """Processing phases for the document pipeline.
    
    Attributes:
        CRAWL: Phase 1 - Crawl and store raw content
        PROCESS: Phase 2 - Process stored content with AI
    """
    CRAWL = "crawl"
    PROCESS = "process"

class DatabaseManager:
    """Database operations manager for the document pipeline.
    
    This class handles all interactions with the Supabase database,
    including storing and updating document chunks.
    
    Attributes:
        client: Initialized Supabase client
    """
    
    def __init__(self, client: Client) -> None:
        """Initialize database manager.
        
        Args:
            client: Supabase client instance
        """
        self.client = client
    
    async def store_page_chunk(self, chunk: Dict[str, Any]) -> None:
        """Store a processed chunk in the database.
        
        Args:
            chunk: Document chunk data including content and metadata
            
        Raises:
            Exception: If there's an error storing the chunk
        """
        try:
            url = chunk.get('url')
            chunk_number = chunk.get('chunk_number')
            
            # Prepare and execute the query
            response = self.client.table(ProcessingConfig.TABLE).upsert(
                chunk, 
                on_conflict='url,chunk_number'
            ).execute()
            
            if hasattr(response, 'error') and response.error:
                raise Exception(f"Supabase error: {response.error}")
            
            logging.info(f"Upserted chunk {chunk_number} for URL {url}")
            
        except Exception as e:
            logging.error(f"Error storing chunk: {str(e)}")
            raise

class ContentProcessor:
    """Content processing operations manager.
    
    This class handles all content processing operations including
    text chunking, embedding generation, and AI-powered summarization.
    
    Attributes:
        openai_client: Optional OpenAI client for AI operations
    """
    
    def __init__(self, openai_client: Optional[openai.AsyncOpenAI] = None) -> None:
        """Initialize content processor.
        
        Args:
            openai_client: Optional OpenAI client for AI operations
        """
        self.openai_client = openai_client

    def chunk_text(
        self, 
        text: str, 
        chunk_size: int = ProcessingConfig.CHUNK_SIZE
    ) -> List[str]:
        """Split text into chunks, respecting content boundaries.
        
        Args:
            text: Text to split into chunks
            chunk_size: Target size for each chunk
            
        Returns:
            List of text chunks
        """
        chunks: List[str] = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size
            
            if end >= text_length:
                chunks.append(text[start:].strip())
                break

            # Try to find natural break points
            chunk = text[start:end]
            
            # Check for code block boundary
            code_block = chunk.rfind('```')
            if code_block != -1 and code_block > chunk_size * 0.3:
                end = start + code_block
            
            # Check for paragraph break
            elif '\n\n' in chunk:
                last_break = chunk.rfind('\n\n')
                if last_break > chunk_size * 0.3:
                    end = start + last_break
            
            # Check for sentence break
            elif '. ' in chunk:
                last_period = chunk.rfind('. ')
                if last_period > chunk_size * 0.3:
                    end = start + last_period + 1

            chunk = text[start:end].strip()
            if chunk and len(chunk) >= ProcessingConfig.MIN_CHUNK_LENGTH:
                chunks.append(chunk)

            start = end

        return chunks

    @retry(
        stop=stop_after_attempt(ProcessingConfig.MAX_RETRIES),
        wait=wait_exponential(
            multiplier=1, 
            min=ProcessingConfig.RETRY_MIN_WAIT, 
            max=ProcessingConfig.RETRY_MAX_WAIT
        ),
        retry=retry_if_exception_type(openai.APIError)
    )
    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding vector for text.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Embedding vector or None on error
            
        Raises:
            ValueError: If OpenAI client is not initialized
        """
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
            
        try:
            response = await self.openai_client.embeddings.create(
                model=config.embedding_model,
                input=text
            )
            if hasattr(response.data[0], 'embedding'):
                return response.data[0].embedding
            else:
                logging.error("Unexpected embedding response format")
                return None
        except Exception as e:
            logging.error(f"Error generating embedding: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(ProcessingConfig.MAX_RETRIES),
        wait=wait_exponential(
            multiplier=1, 
            min=ProcessingConfig.RETRY_MIN_WAIT, 
            max=ProcessingConfig.RETRY_MAX_WAIT
        ),
        retry=retry_if_exception_type(openai.APIError)
    )
    async def get_title_and_summary(
        self, 
        chunk: str, 
        url: str
    ) -> Dict[str, str]:
        """Generate title and summary for content chunk.
        
        Args:
            chunk: Content chunk to process
            url: Source URL of the content
            
        Returns:
            Dictionary containing title and summary
            
        Raises:
            ValueError: If OpenAI client is not initialized
        """
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
            
        try:
            response = await self.openai_client.chat.completions.create(
                model=config.llm_model,
                messages=[
                    {"role": "system", "content": ProcessingConfig.SYSTEM_PROMPT},
                    {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}
                ],
                response_format={"type": "json_object"},
                temperature=config.llm_temperature
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logging.error(f"Error generating title/summary for {url}: {str(e)}")
            raise

class DocumentPipeline:
    """Main document processing pipeline.
    
    This class orchestrates the entire document processing workflow,
    managing both the crawling and AI processing phases.
    
    Attributes:
        supabase: Supabase client
        db: Database manager instance
        processor: Content processor instance
    """
    
    def __init__(self) -> None:
        """Initialize the document pipeline."""
        self.supabase = create_client(
            config.supabase_url,
            config.supabase_service_key
        )
        self.db = DatabaseManager(self.supabase)
        self.processor = ContentProcessor()

    async def crawl_urls(
        self, 
        urls: Sequence[str], 
        max_concurrent: int, 
        source: str
    ) -> None:
        """Crawl URLs and process content.
        
        Args:
            urls: List of URLs to crawl
            max_concurrent: Maximum concurrent crawls
            source: Source identifier for the content
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        self.processor.openai_client = openai.AsyncOpenAI(
            api_key=config.openai_api_key
        )
        
        async def process_url(url: str) -> None:
            async with semaphore:
                try:
                    crawler = AsyncWebCrawler()
                    async with crawler as browser:
                        config = CrawlerRunConfig(cache_mode=CacheMode.ENABLED)
                        result = await browser.arun(url, config)
                        
                        # Extract content from crawler result
                        content = None
                        if result:
                            if hasattr(result, 'markdown'):
                                content = result.markdown if isinstance(result.markdown, str) else getattr(result.markdown, 'str', None)
                            if not content and hasattr(result, 'html'):
                                content = result.html
                            if not content and hasattr(result, 'text'):
                                content = result.text
                        
                        if content:
                            chunks = self.processor.chunk_text(content)
                            for i, chunk_content in enumerate(chunks):
                                title_summary = await self.processor.get_title_and_summary(chunk_content, url)
                                embedding = await self.processor.get_embedding(chunk_content)
                                
                                chunk = {
                                    "url": url,
                                    "chunk_number": i,
                                    "title": title_summary["title"],
                                    "summary": title_summary["summary"],
                                    "content": chunk_content,
                                    "metadata": {
                                        "source": source,
                                        "processed_at": datetime.now(timezone.utc).isoformat(),
                                        "chunk_size": len(chunk_content)
                                    },
                                    "embedding": embedding if embedding else None
                                }
                                await self.db.store_page_chunk(chunk)
                            
                            logging.info(f"Successfully processed {url}")
                        else:
                            logging.error(f"Failed to crawl {url}: No content retrieved")
                except Exception as e:
                    logging.error(f"Error processing {url}: {str(e)}")
        
        tasks = [process_url(url) for url in urls]
        await asyncio.gather(*tasks)

    async def process_stored_pages(self, batch_size: int) -> None:
        """Process stored pages in batches with AI.
        
        Args:
            batch_size: Number of pages to process in each batch
        """
        self.processor.openai_client = openai.AsyncOpenAI(
            api_key=config.openai_api_key
        )
        
        try:
            offset = 0
            total_processed = 0
            
            while True:
                response = self.supabase.table(ProcessingConfig.TABLE) \
                    .select("*") \
                    .is_("embedding", "null") \
                    .range(offset, offset + batch_size - 1) \
                    .execute()
                
                if hasattr(response, 'error') and response.error:
                    raise Exception(f"Supabase error: {response.error}")
                    
                pages = response.data
                if not pages:
                    if total_processed == 0:
                        logging.info("No unprocessed pages found")
                    else:
                        logging.info(f"Completed processing {total_processed} pages")
                    return
                
                batch = pages
                tasks = []
                
                for page in batch:
                    tasks.append(self.processor.get_embedding(page["content"]))
                
                embeddings = await asyncio.gather(*tasks)
                
                for page, embedding in zip(batch, embeddings):
                    self.supabase.table(ProcessingConfig.TABLE).update({
                        "embedding": embedding,
                        "metadata": {
                            **page.get("metadata", {}),
                            "processed_at": datetime.now(timezone.utc).isoformat()
                        }
                    }).eq("id", page["id"]).execute()
                
                logging.info(f"Processed batch of {len(batch)} pages")
                total_processed += len(batch)
                offset += batch_size
                
        except Exception as e:
            logging.error(f"Error processing stored pages: {str(e)}")
            if total_processed > 0:
                logging.info(f"Partially completed: processed {total_processed} pages before error")
            raise

def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed command line arguments
        
    Raises:
        SystemExit: If required arguments are missing
    """
    parser = argparse.ArgumentParser(
        description="Two-phase document processing pipeline for web content",
        usage="%(prog)s {crawl,process} [options]\n\n"
               "Phases:\n"
               "  crawl    Phase 1: Crawl websites and store raw content\n"
               "  process  Phase 2: Process stored content with AI (embeddings & summaries)\n\n"
               "Options:\n"
               "  --source SOURCE           Source identifier for crawled content (required for crawl phase)\n"
               "  --urls URL [URL ...]      List of URLs to crawl (required for crawl phase)\n"
               "  --max-concurrent N        Maximum number of concurrent crawls (default: 5)\n"
               "  --batch-size N            Number of pages to process in each batch (default: 50)\n\n"
               "Examples:\n"
               "  # Crawl phase example:\n"
               "  %(prog)s crawl --source breastcancer_org --urls https://www.breastcancer.org/treatment\n\n"
               "  # Process phase example:\n"
               "  %(prog)s process --batch-size 100\n"
    )
    parser.add_argument(
        "phase",
        type=str,
        choices=['crawl', 'process'],
        metavar="{crawl,process}",
        help="Processing phase to run (crawl or process)"
    )
    parser.add_argument(
        "--source",
        type=str,
        help="Source identifier for crawled content (required for crawl phase)"
    )
    parser.add_argument(
        "--urls",
        nargs="+",
        help="List of URLs to crawl (required for crawl phase)"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=ProcessingConfig.MAX_CONCURRENT,
        help="Maximum concurrent crawls"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=ProcessingConfig.BATCH_SIZE,
        help="Batch size for processing"
    )
    args = parser.parse_args()
    
    # Convert phase string to enum
    args.phase = ProcessingPhase(args.phase)
    
    # Validate required arguments for each phase
    if args.phase == ProcessingPhase.CRAWL:
        if not args.source:
            parser.error("--source is required for crawl phase")
        if not args.urls:
            parser.error("--urls is required for crawl phase")
            
    return args

async def main() -> None:
    """Main entry point for the document processing pipeline."""
    args = parse_args()
    pipeline = DocumentPipeline()
    
    try:
        if args.phase == ProcessingPhase.CRAWL:
            await pipeline.crawl_urls(
                urls=args.urls,
                max_concurrent=args.max_concurrent,
                source=args.source
            )
        else:  # ProcessingPhase.PROCESS
            await pipeline.process_stored_pages(batch_size=args.batch_size)
    except Exception as e:
        logging.error(f"Pipeline error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
