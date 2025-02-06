"""
Supabase Database Record Updater for Mammo.Chat

This module provides functionality to update and fix records in the Supabase database,
specifically focusing on generating and updating titles, summaries, and embeddings
for content chunks that may have failed in previous processing attempts.

Features:
    - Batch processing of database records
    - Automatic retry mechanism for API calls
    - Progress tracking with tqdm
    - Comprehensive logging
    - Configurable settings via Config class

Requirements:
    - OpenAI API key for GPT and embedding models
    - Supabase credentials for database access
    - Environment variables in .env file:
        - OPENAI_API_KEY
        - SUPABASE_URL
        - SUPABASE_SERVICE_KEY

Usage:
    Run the script directly to process all records:
    ```
    python update_supabase.py
    ```
    
    The script will:
    1. Load records from Supabase in batches
    2. Check for records with missing or error states
    3. Generate new titles, summaries, and embeddings as needed
    4. Update the database with new values
"""

import os
import json
import logging
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass

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

@dataclass
class Config:
    """Configuration settings for the update process.
    
    This class centralizes all configuration parameters and makes them easily
    accessible throughout the module. Use class attributes for values that
    should be consistent across all instances.
    """
    
    # OpenAI Model Settings
    LLM_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    # Retry Settings
    MAX_RETRIES: int = 3
    RETRY_MIN_WAIT: int = 4
    RETRY_MAX_WAIT: int = 10
    
    # Batch Processing
    BATCH_SIZE: int = 50
    
    # Database Settings
    TABLE_NAME: str = "site_pages"
    CONTENT_COLUMN: str = "content"
    URL_COLUMN: str = "url"
    TITLE_COLUMN: str = "title"
    SUMMARY_COLUMN: str = "summary"
    EMBEDDING_COLUMN: str = "embedding"
    
    # Logging Settings
    LOG_FILE: str = "update_script.log"
    LOG_FORMAT: str = "%(asctime)s - %(levelname)s - %(message)s"
    
    # Content Processing
    CHUNK_TRUNCATE_LENGTH: int = 1000
    MIN_CHUNK_LENGTH: int = 10
    
    # GPT Prompt
    PROMPT_TEMPLATE: str = """You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative."""

class DatabaseError(Exception):
    """Custom exception for database-related errors."""
    pass

class ProcessingError(Exception):
    """Custom exception for content processing errors."""
    pass

# Initialize logging
def setup_logging() -> None:
    """Configure logging with file and optional console output."""
    logging.basicConfig(
        level=logging.INFO,
        format=Config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(Config.LOG_FILE),
            # Uncomment to enable console logging
            # logging.StreamHandler()
        ]
    )

# Initialize clients
def initialize_clients() -> Tuple[Client, openai.AsyncOpenAI]:
    """Initialize Supabase and OpenAI clients with error handling."""
    try:
        load_dotenv()
        
        supabase = create_client(
            os.getenv("SUPABASE_URL", ""),
            os.getenv("SUPABASE_SERVICE_KEY", "")
        )
        
        openai_client = openai.AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "")
        )
        
        return supabase, openai_client
        
    except Exception as e:
        logging.critical(f"Failed to initialize clients: {str(e)}")
        raise

@retry(
    stop=stop_after_attempt(Config.MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=Config.RETRY_MIN_WAIT, max=Config.RETRY_MAX_WAIT),
    retry=retry_if_exception_type(openai.APIError),
    before_sleep=lambda retry_state: logging.warning(
        f"Retry #{retry_state.attempt_number} for {retry_state.fn.__name__} due to {retry_state.outcome.exception()}"
    )
)
async def get_title_and_summary(chunk: str, url: str) -> Tuple[str, str]:
    """Generate title and summary for a content chunk using GPT.
    
    Args:
        chunk: The content to process
        url: The source URL for context
    
    Returns:
        Tuple of (title, summary)
        
    Raises:
        ProcessingError: If the content processing fails
        openai.APIError: If the API call fails (will trigger retry)
    """
    try:
        clean_chunk = chunk.strip()
        if len(clean_chunk) < Config.MIN_CHUNK_LENGTH:
            logging.warning(f"Skipping short chunk: {len(clean_chunk)} chars")
            return "", ""
            
        logging.info(f"Processing chunk from {url}")
        
        truncated_chunk = clean_chunk[:Config.CHUNK_TRUNCATE_LENGTH]
        if len(clean_chunk) > Config.CHUNK_TRUNCATE_LENGTH:
            logging.debug(f"Truncated chunk from {len(clean_chunk)} to {Config.CHUNK_TRUNCATE_LENGTH} chars")

        completion = await openai_client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=[
                {"role": "system", "content": Config.PROMPT_TEMPLATE},
                {"role": "user", "content": f"URL: {url}\n\nCHUNK: {truncated_chunk}"}
            ],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(completion.choices[0].message.content)
        return (
            result.get(Config.TITLE_COLUMN, "").strip(),
            result.get(Config.SUMMARY_COLUMN, "").strip()
        )
        
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse GPT response: {str(e)}")
        raise ProcessingError("Invalid JSON response from GPT")
    except openai.APIError as e:
        logging.error(f"OpenAI API Error: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in title/summary generation: {str(e)}")
        raise ProcessingError(f"Failed to generate title/summary: {str(e)}")

@retry(
    stop=stop_after_attempt(Config.MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=Config.RETRY_MIN_WAIT, max=Config.RETRY_MAX_WAIT),
    retry=retry_if_exception_type(openai.APIError)
)
async def get_embedding(text: str) -> List[float]:
    """Generate embedding vector for text using OpenAI's API.
    
    Args:
        text: The text to generate embedding for
    
    Returns:
        List of floating point numbers representing the embedding
        
    Raises:
        openai.APIError: If the API call fails (will trigger retry)
        ProcessingError: For other processing errors
    """
    try:
        logging.info(f"Generating embedding for: {text[:50]}...")
        response = await openai_client.embeddings.create(
            input=text,
            model=Config.EMBEDDING_MODEL
        )
        return response.data[0].embedding
    except openai.APIError as e:
        logging.error(f"Embedding API Error: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in embedding generation: {str(e)}")
        raise ProcessingError(f"Failed to generate embedding: {str(e)}")

async def process_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process a single database record and determine needed updates.
    
    Args:
        record: The database record to process
        
    Returns:
        Dictionary of updates to apply, or None if no updates needed
    """
    updates: Dict[str, Any] = {}
    
    chunk = record.get(Config.CONTENT_COLUMN, "")
    url = record.get(Config.URL_COLUMN, "")
    current_title = record.get(Config.TITLE_COLUMN, "")
    current_summary = record.get(Config.SUMMARY_COLUMN, "")
    
    try:
        current_embedding = json.loads(record.get(Config.EMBEDDING_COLUMN, "[]"))
    except json.JSONDecodeError:
        current_embedding = []
        logging.warning(f"Invalid embedding format for record {record['id']}")

    # Check if updates are needed
    needs_update = (
        current_title.lower().startswith("error") or 
        (current_summary and current_summary.lower().startswith("error"))
    )

    if needs_update and (chunk or url):
        try:
            new_title, new_summary = await get_title_and_summary(chunk, url)
            
            if new_title and new_title != current_title:
                updates[Config.TITLE_COLUMN] = new_title
                
            if new_summary and new_summary != current_summary:
                updates[Config.SUMMARY_COLUMN] = new_summary

        except ProcessingError as e:
            logging.error(f"Failed to process record {record['id']}: {str(e)}")
            return None

    # Update embedding if needed
    if all(v == 0 for v in current_embedding) and updates.get(Config.TITLE_COLUMN):
        try:
            updates[Config.EMBEDDING_COLUMN] = await get_embedding(
                updates.get(Config.TITLE_COLUMN, current_title)
            )
        except ProcessingError as e:
            logging.error(f"Failed to generate embedding for record {record['id']}: {str(e)}")
            
    return updates if updates else None

async def update_records() -> None:
    """Main update workflow with batched processing.
    
    This function:
    1. Queries the total count of records
    2. Processes records in batches
    3. Updates records that need fixing
    4. Tracks progress and handles errors
    """
    logger.info("Starting update process")
    
    try:
        # Get total record count
        count_result = supabase.table(Config.TABLE_NAME) \
            .select("*", count="exact") \
            .execute()
        total_count = count_result.count
        
        if not total_count:
            logger.info("No records to process")
            return

        total_batches = (total_count + Config.BATCH_SIZE - 1) // Config.BATCH_SIZE
        logger.info(f"Processing {total_count} records in {total_batches} batches")

        with tqdm(total=total_count, desc="Processing records", unit="rec") as pbar:
            for batch_num in range(total_batches):
                offset = batch_num * Config.BATCH_SIZE
                
                try:
                    # Fetch batch of records
                    response = supabase.table(Config.TABLE_NAME) \
                        .select(f"id,{Config.TITLE_COLUMN},{Config.SUMMARY_COLUMN}," \
                               f"{Config.CONTENT_COLUMN},{Config.URL_COLUMN},{Config.EMBEDDING_COLUMN}") \
                        .range(offset, offset + Config.BATCH_SIZE - 1) \
                        .execute()
                    batch = response.data
                    
                    # Process each record
                    for record in batch:
                        updates = await process_record(record)
                        
                        if updates:
                            try:
                                supabase.table(Config.TABLE_NAME) \
                                    .update(updates) \
                                    .eq("id", record["id"]) \
                                    .execute()
                                logger.info(f"Updated record {record['id']}: {list(updates.keys())}")
                            except Exception as e:
                                logger.error(f"Failed to update record {record['id']}: {str(e)}")
                                
                    pbar.update(len(batch))
                    
                except Exception as e:
                    logger.error(f"Failed to process batch {batch_num + 1}: {str(e)}")
                    pbar.update(Config.BATCH_SIZE)
                    continue

        logger.info("Update process completed successfully")
        
    except Exception as e:
        logger.error(f"Update process failed: {str(e)}")
        raise DatabaseError(f"Failed to complete update process: {str(e)}")

# Initialize globals
logger = logging.getLogger(__name__)
setup_logging()
supabase, openai_client = initialize_clients()

if __name__ == "__main__":
    logger.info("Script started")
    try:
        import asyncio
        asyncio.run(update_records())
    except Exception as e:
        logger.critical(f"Critical failure: {str(e)}", exc_info=True)
    finally:
        logger.info("Script finished")