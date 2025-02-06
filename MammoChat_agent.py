"""
MammoChat Agent Module

This module implements an AI-powered medical information assistant specialized in breast cancer information.
It uses RAG (Retrieval Augmented Generation) to provide accurate information from trusted sources.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncGenerator, List, Optional, Dict, Any
import json

from openai import AsyncOpenAI
from supabase import Client
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from config import config

@dataclass
class SourceDeps:
    """
    Source dependencies required for the agent.

    Attributes:
        supabase: Supabase client for database operations
        openai_client: AsyncOpenAI client for AI operations
    """
    supabase: Client
    openai_client: AsyncOpenAI

SYSTEM_PROMPT = """
You are a specialized AI medical assistant whose main role is to provide accurate, 
evidence-based information about breast cancer from trusted sources to help reduce 
patient anxiety.

Key Guidelines:
1. Only use information from BreastCancer.org and Komen.org
2. Always cite your sources with direct links
3. Use clear, patient-friendly language
4. Acknowledge emotional concerns while staying factual
5. Never provide medical advice - direct users to consult healthcare providers
6. If unsure, admit uncertainty and suggest consulting medical professionals

Your responses should:
- Be clear and concise
- Include relevant source links
- Avoid medical jargon when possible
- Emphasize the importance of professional medical consultation
"""

class MammoChat_agent(Agent):
    """
    AI agent specialized in breast cancer information retrieval and communication.
    
    This agent uses RAG to provide accurate information from trusted medical sources
    while maintaining a compassionate and clear communication style.
    """

    async def search_documents(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for relevant documents using semantic similarity.

        Args:
            query: The search query string

        Returns:
            List of relevant documents with their metadata
        """
        # Generate embedding for the query
        response = await self.deps.openai_client.embeddings.create(
            model=config.embedding_model,
            input=query
        )
        query_embedding = response.data[0].embedding

        # Search Supabase for similar documents
        response = self.deps.supabase.rpc(
            'match_documents',
            {'query_embedding': query_embedding, 'match_threshold': 0.5, 'match_count': 5}
        ).execute()

        return response.data if response.data else []

    async def process_query(self, query: str) -> AsyncGenerator[str, None]:
        """
        Process a user query and generate a response using relevant documents.

        Args:
            query: The user's question or query

        Yields:
            Chunks of the generated response
        """
        # Search for relevant documents
        documents = await self.search_documents(query)
        
        if not documents:
            yield ("I apologize, but I couldn't find specific information about that in my trusted sources. "
                  "Please consult with your healthcare provider for accurate medical guidance.")
            return

        # Prepare context from documents
        context = "\n\n".join(f"Source: {doc['url']}\n{doc['content']}" for doc in documents)

        # Generate response using the context
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]

        async for chunk in self.deps.openai_client.chat.completions.create(
            model=config.llm_model,
            messages=messages,
            temperature=0.7,
            stream=True
        ):
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    @classmethod
    async def run(cls, query: str, deps: SourceDeps, **kwargs) -> str:
        """
        Run the agent to process a query and return a complete response.

        Args:
            query: The user's question
            deps: Required dependencies for the agent
            **kwargs: Additional arguments

        Returns:
            Complete response string
        """
        agent = cls(deps=deps)
        response = ""
        async for chunk in agent.process_query(query):
            response += chunk
        return response