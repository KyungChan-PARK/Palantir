# llm_caller.py
"""
LLM caller tool for codebase knowledge analysis.
Adapted from PocketFlow-Tutorial-Codebase-Knowledge.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from ....mcp_init import mcp

# Configure logging
log_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'logs')
os.makedirs(log_directory, exist_ok=True)
log_file = os.path.join(
    log_directory, f"llm_calls_{datetime.now().strftime('%Y%m%d')}.log"
)

# Set up logger
logger = logging.getLogger("llm_logger")
logger.setLevel(logging.INFO)
logger.propagate = False  # Prevent propagation to root logger
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
logger.addHandler(file_handler)

# Simple cache configuration
cache_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "llm_cache.json")

@mcp.tool(
    name="call_llm",
    description="Call LLM API with caching capability for code analysis tasks",
    tags=["llm", "api", "code", "analysis"]
)
async def call_llm(prompt: str, use_cache: bool = True) -> str:
    """
    Call a Large Language Model API with caching capability.
    
    Args:
        prompt (str): The prompt to send to the LLM
        use_cache (bool): Whether to use caching for the response
        
    Returns:
        str: The LLM response text
    """
    # Log the prompt
    logger.info(f"PROMPT: {prompt}")

    # Check cache if enabled
    if use_cache:
        # Load cache from disk
        cache = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    cache = json.load(f)
            except:
                logger.warning(f"Failed to load cache, starting with empty cache")

        # Return from cache if exists
        if prompt in cache:
            logger.info(f"RESPONSE (cached): {cache[prompt][:100]}...")
            return cache[prompt]
    
    # For this implementation, we're using the Claude API directly
    # This can be replaced with any other LLM API
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
        
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=4000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        response_text = response.content[0].text
    except ImportError:
        # Fallback to a simple response if Anthropic is not installed
        logger.warning("Anthropic client not installed. Using fallback response method.")
        # In a real implementation, we would add alternative LLM clients here
        response_text = "Error: Unable to make LLM API call. The Anthropic package is not installed."

    # Log the response
    logger.info(f"RESPONSE: {response_text[:100]}...")

    # Update cache if enabled
    if use_cache:
        # Load cache again to avoid overwrites
        cache = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    cache = json.load(f)
            except:
                pass

        # Add to cache and save
        cache[prompt] = response_text
        try:
            with open(cache_file, "w") as f:
                json.dump(cache, f)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    return response_text
