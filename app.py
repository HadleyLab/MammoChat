"""MammoChat web application.

This module implements the main web interface for the MammoChat application using Streamlit.
It provides an interactive chat interface that allows users to ask questions about breast cancer
and receive reliable information from trusted medical sources.

The application uses a RAG (Retrieval-Augmented Generation) system to ensure accurate
information delivery from reputable sources like BreastCancer.org and Komen.org.
"""

from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import streamlit as st
import json
import logfire
from supabase import Client
from openai import AsyncOpenAI

import os
from config import config, TRUSTED_SOURCES

try:
    # Set OpenAI API key in environment variable
    os.environ["OPENAI_API_KEY"] = config.openai_api_key
    st.write("OpenAI API key configured successfully")
except Exception as e:
    st.error(f"Failed to configure OpenAI API key: {str(e)}")
    logfire.error("OpenAI API key configuration failed", error=str(e))

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
from agent import chat_agent, SourceDeps

try:
    openai_client = AsyncOpenAI()  # Will use OPENAI_API_KEY from environment
    st.write("OpenAI client initialized successfully")
except Exception as e:
    st.error(f"Failed to initialize OpenAI client: {str(e)}")
    logfire.error("OpenAI client initialization failed", error=str(e))

try:
    supabase: Client = Client(
        config.supabase_url,
        config.supabase_service_key
    )
    st.write("Supabase client initialized successfully")
except Exception as e:
    st.error(f"Failed to initialize Supabase client: {str(e)}")
    logfire.error("Supabase client initialization failed", error=str(e))

# Enable logging in cloud environment
if 'STREAMLIT_CLOUD' in os.environ:
    logfire.configure(send_to_logfire='always')
else:
    logfire.configure(send_to_logfire='never')

class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""

    role: Literal['user', 'model']
    timestamp: str
    content: str


def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    Customize how you display system prompts, user prompts,
    tool calls, tool returns, etc.
    """
    # system-prompt
    if part.part_kind == 'system-prompt':
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    # user-prompt
    elif part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    # text
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content)          


async def run_agent_with_streaming(user_input: str):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    """
    # Prepare dependencies
    deps = SourceDeps(
        supabase=supabase,
        openai_client=openai_client
    )

    # Run the agent in a stream
    async with chat_agent.run_stream(
        user_input,
        deps=deps,
        message_history= st.session_state.messages[:-1],  # pass entire conversation so far
    ) as result:
        # We'll gather partial text to show incrementally
        partial_text = ""
        message_placeholder = st.empty()

        # Render partial text as it arrives
        async for chunk in result.stream_text(delta=True):
            partial_text += chunk
            message_placeholder.markdown(partial_text)

        # Now that the stream is finished, we have a final result.
        # Add new messages from this run, excluding user-prompt messages
        filtered_messages = [msg for msg in result.new_messages() 
                            if not (hasattr(msg, 'parts') and 
                                    any(part.part_kind == 'user-prompt' for part in msg.parts))]
        st.session_state.messages.extend(filtered_messages)

        # Add the final response to the messages
        st.session_state.messages.append(
            ModelResponse(parts=[TextPart(content=partial_text)])
        )


async def main():
    st.title("Mammo.Chat™ -- Breast Cancer AI")
    st.write(f"""
    AI-Guided Navigation of Breast Cancer from Trusted Sources:
    {', '.join(TRUSTED_SOURCES)}
    """)

    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display all messages from the conversation so far
    # Each message is either a ModelRequest or ModelResponse.
    # We iterate over their parts to decide how to display them.
    for msg in st.session_state.messages:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)

    # Chat input for the user
    user_input = st.chat_input("What questions do you have about breast cancer?")

    if user_input:
        # We append a new request to the conversation explicitly
        st.session_state.messages.append(
            ModelRequest(parts=[UserPromptPart(content=user_input)])
        )
        
        # Display user prompt in the UI
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display the assistant's partial response while streaming
        with st.chat_message("assistant"):
            # Actually run the agent now, streaming the text
            await run_agent_with_streaming(user_input)


if __name__ == "__main__":
    asyncio.run(main())
