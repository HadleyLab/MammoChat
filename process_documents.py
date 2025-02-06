"""
Document Processing Pipeline for Mammo.Chat

This module provides a comprehensive pipeline for processing documents in two phases:
1. Crawling & Storage Phase: Crawl websites and store raw content without AI processing
2. AI Processing Phase: Process stored content with OpenAI APIs for embeddings and summaries

Features:
    - Two-phase processing to separate crawling from AI processing
    - Configurable crawling parameters
    - Batch processing with progress tracking
    - Comprehensive logging
    - Error handling and retry mechanisms
    - Support for different content sources
    - Memory efficient processing

Requirements:
    - OpenAI API key (only for AI processing phase)
    - Supabase credentials
    - Environment variables in .env file:
        - SUPABASE_URL
        - SUPABASE_SERVICE_KEY
        - OPENAI_API_KEY (only for AI processing)

Usage:
    # Phase 1: Crawl and store raw content
    python process_documents.py crawl --source <source_name> --urls <url1> [<url2> ...] [--max-concurrent 5]
    
    Example:
    python process_documents.py crawl --source breastcancer_org --urls https://www.breastcancer.org/treatment https://www.breastcancer.org/symptoms
    
    # Phase 2: Process stored content with AI
    python process_documents.py process [--batch-size 50]
"""

import os
import sys
import json
import asyncio
import logging
import argparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

# Third-party imports
from dotenv import load_dotenv
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

# Load environment variables
load_dotenv()

@dataclass
class Config:
    """Configuration settings for the document processing pipeline"""
    
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
    
    # OpenAI Settings
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4-mini")
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    MAX_PROMPT_LENGTH: int = 1000
    TEMPERATURE: float = 0.0
    TOP_P: float = 1.0
    
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
    level=Config.LOG_LEVEL,
    format=Config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(Config.LOG_FILE),
        logging.StreamHandler()
    ]
)

class ProcessingPhase(Enum):
    """Enum for different processing phases"""
    CRAWL = "crawl"
    PROCESS = "process"

class DatabaseManager:
    """Handles all database operations"""
    
    def __init__(self, client: Client):
        self.client = client
    
    async def store_page_chunk(self, chunk: Dict[str, Any]) -> None:
        """Store a processed chunk in the site_pages table"""
        try:
            # Use upsert operation with url and chunk_number as unique constraints
            url = chunk.get('url')
            chunk_number = chunk.get('chunk_number')
            
            # Prepare and execute the query synchronously since Supabase client doesn't support async
            response = self.client.table(Config.TABLE).upsert(chunk, on_conflict='url,chunk_number').execute()
            
            if hasattr(response, 'error') and response.error:
                raise Exception(f"Supabase error: {response.error}")
            
            # Log success
            logging.info(f"Upserted chunk {chunk_number} for URL {url}")
            
        except Exception as e:
            logging.error(f"Error storing chunk: {str(e)}")
            raise

class ContentProcessor:
    """Handles content processing operations"""
    
    def __init__(self, openai_client: Optional[openai.AsyncOpenAI] = None):
        self.openai_client = openai_client

    def chunk_text(self, text: str, chunk_size: int = Config.CHUNK_SIZE) -> List[str]:
        """Split text into chunks, respecting content boundaries"""
        chunks = []
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
            if chunk and len(chunk) >= Config.MIN_CHUNK_LENGTH:
                chunks.append(chunk)

            start = end

        return chunks

    @retry(
        stop=stop_after_attempt(Config.MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=Config.RETRY_MIN_WAIT, max=Config.RETRY_MAX_WAIT),
        retry=retry_if_exception_type(openai.APIError)
    )
    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding vector for text"""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
            
        try:
            response = await self.openai_client.embeddings.create(
                model=Config.EMBEDDING_MODEL,
                input=text
            )
            # Properly handle the embedding response
            if hasattr(response.data[0], 'embedding'):
                return response.data[0].embedding
            else:
                logging.error("Unexpected embedding response format")
                return None
        except Exception as e:
            logging.error(f"Error generating embedding: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(Config.MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=Config.RETRY_MIN_WAIT, max=Config.RETRY_MAX_WAIT),
        retry=retry_if_exception_type(openai.APIError)
    )
    async def get_title_and_summary(self, chunk: str, url: str) -> Dict[str, str]:
        """Generate title and summary for content chunk"""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
            
        system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
        Return a JSON object with 'title' and 'summary' keys.
        For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
        For the summary: Create a concise summary of the main points in this chunk.
        Keep both title and summary concise but informative."""
        
        try:
            response = await self.openai_client.chat.completions.create(
                model=Config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}
                ],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logging.error(f"Error generating title/summary for {url}: {str(e)}")
            raise

class DocumentPipeline:
    """Main pipeline for document processing"""
    
    def __init__(self):
        # Initialize clients
        self.supabase = create_client(
            os.getenv("SUPABASE_URL", ""),
            os.getenv("SUPABASE_SERVICE_KEY", "")
        )
        self.db = DatabaseManager(self.supabase)
        self.processor = ContentProcessor()

    async def crawl_urls(self, urls: List[str], max_concurrent: int, source: str) -> None:
        """Crawl URLs and process content directly into site_pages"""
        semaphore = asyncio.Semaphore(max_concurrent)
        # Initialize OpenAI client for processing
        self.processor.openai_client = openai.AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "")
        )
        
        async def process_url(url: str):
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
                            # Split content into chunks and process each
                            chunks = self.processor.chunk_text(content)
                            for i, chunk_content in enumerate(chunks):
                                # Generate title and summary
                                title_summary = await self.processor.get_title_and_summary(chunk_content, url)
                                
                                # Generate embedding
                                embedding = await self.processor.get_embedding(chunk_content)
                                
                                # Store chunk
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
        """Process stored pages in batches with AI"""
        # Initialize OpenAI client for processing
        self.processor.openai_client = openai.AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "")
        )
        
        try:
            offset = 0
            total_processed = 0
            
            while True:
                # Get unprocessed pages with pagination
                response = self.supabase.table(Config.TABLE) \
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
                
                # Process current batch
                batch = pages
                tasks = []
                
                for page in batch:
                    # Generate embedding
                    tasks.append(self.processor.get_embedding(page["content"]))
                
                # Wait for all embeddings in batch
                embeddings = await asyncio.gather(*tasks)
                
                # Update pages with embeddings
                for page, embedding in zip(batch, embeddings):
                    self.supabase.table(Config.TABLE).update({
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

def parse_args():
    """Parse command line arguments"""
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
        default=Config.MAX_CONCURRENT,
        help="Maximum concurrent crawls"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=Config.BATCH_SIZE,
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

async def main():
    """Main entry point"""
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
