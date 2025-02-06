"""
MammoChat - Breast Cancer Information Assistant

A Streamlit-based chat interface that provides reliable breast cancer information 
from trusted sources like BreastCancer.org and Komen.org using AI-powered search and retrieval.
"""

from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os

import streamlit as st
import logfire
from supabase import Client
from openai import AsyncOpenAI
from dotenv import load_dotenv

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter
)
from MammoChat_agent import MammoChat_agent, SourceDeps
from config import config

# Load environment variables
load_dotenv()

# Initialize clients
openai_client = AsyncOpenAI(api_key=config.openai_api_key)
supabase: Client = Client(
    config.supabase_url,
    config.supabase_service_key
)

# Configure logging
logfire.configure(send_to_logfire='never')

class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""
    role: Literal['user', 'model']
    timestamp: str
    content: str

def display_message_part(part) -> None:
    """
    Display a single part of a message in the Streamlit UI.

    Args:
        part: Message part object containing content and type information
    """
    match part.part_kind:
        case 'system-prompt':
            with st.chat_message("system"):
                st.markdown(f"**System**: {part.content}")
        case 'user-prompt':
            with st.chat_message("user"):
                st.markdown(part.content)
        case 'text':
            with st.chat_message("assistant"):
                st.markdown(part.content)

async def run_agent_with_streaming(user_input: str) -> None:
    """
    Run the agent with streaming text response while maintaining conversation history.

    Args:
        user_input: The user's input prompt
    """
    # Prepare dependencies
    deps = SourceDeps(
        supabase=supabase,
        openai_client=openai_client
    )

    # Run the agent in streaming mode
    async with MammoChat_agent.run_stream(
        user_input,
        deps=deps,
        message_history=st.session_state.messages[:-1],
    ) as result:
        partial_text = ""
        message_placeholder = st.empty()

        # Stream partial responses
        async for chunk in result.stream_text(delta=True):
            partial_text += chunk
            message_placeholder.markdown(partial_text)

        # Update conversation history
        filtered_messages = [
            msg for msg in result.new_messages() 
            if not (hasattr(msg, 'parts') and 
                   any(part.part_kind == 'user-prompt' for part in msg.parts))
        ]
        st.session_state.messages.extend(filtered_messages)
        
        # Add final response
        st.session_state.messages.append(
            ModelResponse(parts=[TextPart(content=partial_text)])
        )

async def main() -> None:
    """Initialize and run the main Streamlit application."""
    st.title("Mammo.Chat™ -- Breast Cancer AI")
    st.write("""
    AI-Guided Navigation of Breast Cancer from Trusted Sources:
    [BreastCancer.org](https://www.breastcancer.org), [Komen.org](https://www.komen.org)
    """)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display conversation history
    for msg in st.session_state.messages:
        if isinstance(msg, (ModelRequest, ModelResponse)):
            for part in msg.parts:
                display_message_part(part)

    # Handle user input
    if user_input := st.chat_input("What questions do you have about breast cancer?"):
        # Add user message to history
        st.session_state.messages.append(
            ModelRequest(parts=[UserPromptPart(content=user_input)])
        )
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Process and display assistant response
        with st.chat_message("assistant"):
            await run_agent_with_streaming(user_input)

if __name__ == "__main__":
    asyncio.run(main())
