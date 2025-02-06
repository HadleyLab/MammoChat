"""Configuration module for MammoChat application.

This module provides centralized configuration management for the MammoChat application,
including environment variables, model settings, and search parameters.

Attributes:
    EMBEDDING_MODEL (str): Default model for generating embeddings
    LLM_MODEL (str): Default language model for text generation
    LLM_TEMPERATURE (float): Temperature setting for language model responses
    MATCH_THRESHOLD (float): Minimum similarity threshold for content matching
    MATCH_COUNT (int): Number of matches to return in search results
    TRUSTED_SOURCES (list): List of approved medical information sources
"""

import os
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv
import streamlit as st

# Load environment variables as fallback
load_dotenv()

def get_secret(key: str, default: str = '') -> str:
    """Get secret from Streamlit secrets or environment variables.
    
    Args:
        key: The key to look up
        default: Default value if neither source has the key
        
    Returns:
        The secret value
    """
    try:
        return st.secrets["api_keys"][key]
    except (FileNotFoundError, KeyError):
        # Fall back to environment variables if Streamlit secrets not available
        env_key = key.upper()
        if key == 'openai':
            env_key = 'OPENAI_API_KEY'
        elif key == 'supabase_url':
            env_key = 'SUPABASE_URL'
        elif key == 'supabase_service_key':
            env_key = 'SUPABASE_SERVICE_KEY'
        return os.getenv(env_key, default)

# Model configurations
EMBEDDING_MODEL: str = "text-embedding-3-small"
LLM_MODEL: str = "gpt-4o-mini"
LLM_TEMPERATURE: float = 0.7
MATCH_THRESHOLD: float = 0.5
MATCH_COUNT: int = 5

# Trusted sources
TRUSTED_SOURCES: List[str] = [
    "BreastCancer.org",
    "Komen.org"
]

@dataclass
class Config:
    """Application configuration settings.
    
    This class encapsulates all configuration settings for the MammoChat application,
    including API credentials, model parameters, and search settings.
    
    Attributes:
        openai_api_key (str): OpenAI API key for model access
        supabase_url (str): Supabase instance URL
        supabase_service_key (str): Supabase service role API key
        embedding_model (str): Model used for generating embeddings
        llm_model (str): Language model for text generation
        llm_temperature (float): Temperature parameter for response generation
        match_threshold (float): Minimum similarity score for content matching
        match_count (int): Number of relevant matches to return
    """
    # API Keys (from Streamlit secrets or environment variables)
    openai_api_key: str = get_secret('openai')
    supabase_url: str = get_secret('supabase_url')
    supabase_service_key: str = get_secret('supabase_service_key')
    
    # Model settings
    embedding_model: str = EMBEDDING_MODEL
    llm_model: str = LLM_MODEL
    llm_temperature: float = LLM_TEMPERATURE
    
    # Search settings
    match_threshold: float = MATCH_THRESHOLD
    match_count: int = MATCH_COUNT

    def __post_init__(self) -> None:
        """Validate configuration after initialization.
        
        Raises:
            ValueError: If any required API keys are missing
        """
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
        if not self.supabase_url:
            raise ValueError("Supabase URL is required")
        if not self.supabase_service_key:
            raise ValueError("Supabase service key is required")

# Create config instance
config = Config()
