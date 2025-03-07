"""MammoChat agent module.

This module implements the core agent functionality for the MammoChat application,
providing tools for retrieving and processing medical information from trusted sources.
It uses a RAG (Retrieval-Augmented Generation) system to ensure accurate information
delivery from reputable sources like BreastCancer.org and Komen.org.

The agent is configured to provide empathetic, accurate responses with proper citations
and source attribution for all medical information.
"""

from __future__ import annotations as _annotations
from dataclasses import dataclass
import logfire
import asyncio
import os
from typing import List, Optional
import streamlit as st

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client

from config import config, TRUSTED_SOURCES

# Ensure OpenAI API key is set in environment
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = config.openai_api_key

# Format trusted sources for system prompt
sources_list = "\n".join(f"{i+1}. {source}" for i, source in enumerate(TRUSTED_SOURCES))

# Initialize OpenAI model
model = OpenAIModel(config.llm_model)

# Configure logging with more verbose error reporting
logfire.configure(send_to_logfire='always')
logfire.info("Agent module initializing")

@dataclass
class SourceDeps:
    """Dependencies required for agent operation.
    
    Attributes:
        supabase: Client for database operations
        openai_client: Client for OpenAI API calls
    """
    supabase: Client
    openai_client: AsyncOpenAI

# System prompt defining agent behavior and response format
system_prompt = f"""
**Role:** Medical Information Specialist  

**Sources:** Only use:  
{sources_list}

**Protocols:**  
- Do NOT cite any other sources or widely known facts and public knowledge 
- Every factual statement MUST have numbered citations¹ from trusted sources only
- Automatic footnotes with full URLs for all citations  
- No rhetorical questions - provide sources immediately
"""

# Initialize the agent with configuration
chat_agent = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=SourceDeps,
    retries=2
)

async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Generate embedding vector for text using OpenAI's API.
    
    Args:
        text: The text to generate embeddings for
        openai_client: Initialized OpenAI client
        
    Returns:
        List[float]: The embedding vector, or zero vector on error
    """
    try:
        response = await openai_client.embeddings.create(
            model=config.embedding_model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logfire.error("Error generating embedding", error=str(e))
        st.error(f"Failed to generate embedding: {str(e)}")
        return [0] * 1536  # Return zero vector on error

@chat_agent.tool
async def retrieve_relevant_documentation(ctx: RunContext[SourceDeps], user_query: str) -> str:
    """Retrieve relevant documentation chunks based on the query using RAG.
    
    Uses semantic search to find the most relevant documentation chunks that match
    the user's query. The search is performed using embedding similarity.
    
    Args:
        ctx: Context containing Supabase and OpenAI clients
        user_query: The user's question or query
        
    Returns:
        str: Formatted string containing relevant documentation chunks
    """
    try:
        # Get the embedding for the query
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        
        try:
            # Query Supabase for relevant documents
            logfire.info("Attempting to match documents with query embedding")
            result = ctx.deps.supabase.rpc(
                'match_site_pages',
                {
                    'query_embedding': query_embedding,
                    'match_count': config.match_count
                }
            ).execute()
            
            if not result.data:
                logfire.warning("No matching documents found in database")
                return "No relevant documentation found."
                
            logfire.info(f"Found {len(result.data)} matching documents")
        except Exception as e:
            error_msg = f"Error querying Supabase: {str(e)}"
            logfire.error(error_msg)
            st.error(error_msg)
            return error_msg
            
        # Format the results
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
            # {doc['title']}
            {doc['content']}
            """
            formatted_chunks.append(chunk_text)
            
        # Join all chunks with a separator
        return "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        error_msg = f"Error retrieving documentation: {str(e)}"
        logfire.error(error_msg)
        st.error(error_msg)
        return error_msg

@chat_agent.tool
async def list_documentation_pages(ctx: RunContext[SourceDeps]) -> List[str]:
    """Retrieve a list of all available documentation pages.
    
    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    try:
        # Query Supabase for unique URLs
        result = ctx.deps.supabase.from_('site_pages') \
            .select('url') \
            .execute()

        if not result.data:
            return []
            
        # Extract unique URLs
        urls = sorted(set(doc['url'] for doc in result.data))
        return urls
        
    except Exception as e:
        error_msg = f"Error retrieving documentation pages: {str(e)}"
        logfire.error(error_msg)
        st.error(error_msg)
        return []

@chat_agent.tool
async def get_page_content(ctx: RunContext[SourceDeps], url: str) -> str:
    """Retrieve the full content of a specific documentation page.
    
    Combines all chunks of a page in order to reconstruct the complete content.
    
    Args:
        ctx: Context containing the Supabase client
        url: URL of the page to retrieve
        
    Returns:
        str: Complete page content with all chunks combined in order
    """
    try:
        # Query Supabase for all chunks of this URL, ordered by chunk_number
        result = ctx.deps.supabase.from_('site_pages') \
            .select('title, content, chunk_number') \
            .eq('url', url) \
            .order('chunk_number') \
            .execute()
        
        if not result.data:
            return f"No content found for URL: {url}"
            
        # Format the page with its title and all chunks
        page_title = result.data[0]['title'].split(' - ')[0]  # Get the main title
        formatted_content = [f"# {page_title}\n"]
        
        # Add each chunk's content
        for chunk in result.data:
            formatted_content.append(chunk['content'])
            
        # Join everything together
        return "\n\n".join(formatted_content)
        
    except Exception as e:
        error_msg = f"Error retrieving page content: {str(e)}"
        logfire.error(error_msg)
        st.error(error_msg)
        return error_msg
