#!/usr/bin/env python3
"""
Shared utilities for LLM calls with logging.
"""

import logging
import os
from datetime import datetime
from openai import OpenAI
from typing import Optional

# Default folders (relative to project root)
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def ensure_directories():
    """Create prompts/ and results/ directories if they don't exist."""
    os.makedirs(PROMPTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def setup_logger(log_file_path):
    """Set up a logger that writes to both file and console."""
    logger = logging.getLogger('llm_queries')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # File handler - detailed logging
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger


def call_gpt(prompt, model, temperature=0.7, logger=None, query_id=None, use_reasoning=False, reasoning_effort="high", base_url=None, api_key=None):
    """
    Call GPT model with the given prompt and log the query.
    
    Args:
        prompt: The prompt text to send
        model: GPT model to use (or model name for Modal endpoint)
        temperature: Temperature for generation
        logger: Optional logger instance
        query_id: Optional identifier for this query (e.g., "Trial1_Agent5")
        use_reasoning: If True, use the reasoning API (client.responses.create)
        reasoning_effort: Reasoning effort level ("none", "low", "medium", "high")
        base_url: Optional base URL for OpenAI-compatible API (e.g., Modal endpoint)
        api_key: Optional API key (defaults to OPENAI_API_KEY env var)
    
    Returns:
        Response content from GPT
    """
    # Support Modal or other OpenAI-compatible endpoints
    if base_url:
        client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")
    else:
        client = OpenAI(api_key=api_key)
    
    # Log the query
    if logger:
        query_label = f"Query {query_id}" if query_id else "Query"
        logger.info(f"{'='*80}")
        logger.info(f"{query_label}")
        logger.info(f"{'='*80}")
        logger.info(f"Model: {model}")
        logger.info(f"Temperature: {temperature}")
        if use_reasoning:
            logger.info(f"Reasoning effort: {reasoning_effort}")
        logger.info(f"\nPROMPT:\n{prompt}")
        logger.info(f"\n{'='*80}\n")
    
    try:
        if use_reasoning:
            # Use reasoning API
            response = client.responses.create(
                model=model,
                input=prompt,
                reasoning={
                    "effort": reasoning_effort
                }
            )
            # Extract the response - reasoning API returns a Response object with output attribute
            # Response structure: response.output = [ResponseReasoningItem, ResponseOutputMessage, ...]
            # The output message has content list with ResponseOutputText items containing text
            result = None
            output_list = getattr(response, 'output', None)
            if output_list and isinstance(output_list, list):
                for item in output_list:
                    # Look for message type items (output messages)
                    item_type = getattr(item, 'type', None)
                    if item_type == 'message':
                        content = getattr(item, 'content', None)
                        if isinstance(content, list) and len(content) > 0:
                            # Get text from first content item
                            first_content = content[0]
                            text = getattr(first_content, 'text', None)
                            if text:
                                result = text
                                break
            # Fallback to string representation if extraction failed
            if result is None:
                result = str(response)
        else:
            # Use standard chat completions API
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            result = response.choices[0].message.content
        
        # Log the response
        if logger:
            logger.info(f"RESPONSE:\n{result}")
            logger.info(f"{'='*80}\n")
        
        return result
    except Exception as e:
        error_msg = f"Error calling GPT API: {e}"
        if logger:
            logger.error(f"ERROR: {error_msg}")
        raise Exception(error_msg)


def run_trials(prompt_text, model, temperature, num_trials, output_base, logger=None, use_reasoning=False, reasoning_effort="high"):
    """
    Run multiple trials and save all outputs.
    
    Args:
        prompt_text: The prompt to use
        model: GPT model to use
        temperature: Temperature for generation
        num_trials: Number of trials to run
        output_base: Base name for output files
        logger: Optional logger instance
        use_reasoning: If True, use the reasoning API
        reasoning_effort: Reasoning effort level ("none", "low", "medium", "high")
    
    Returns:
        List of outputs
    """
    outputs = []
    
    print(f"Running {num_trials} trials...")
    
    for trial in range(1, num_trials + 1):
        print(f"\nTrial {trial}/{num_trials}...")
        query_id = f"Trial{trial}"
        try:
            output = call_gpt(prompt_text, model, temperature, logger, query_id, use_reasoning, reasoning_effort)
            outputs.append(output)
        except Exception as e:
            print(f"Error in trial {trial}: {e}")
            outputs.append(f"[ERROR: {e}]")
    
    # Ensure results directory exists
    ensure_directories()
    
    # Save output file
    if num_trials > 1:
        # Multiple trials - save as all_trials.txt
        output_file = os.path.join(RESULTS_DIR, f"{output_base}_all_trials.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Results from {num_trials} trials\n")
            f.write("="*80 + "\n\n")
            for i, output in enumerate(outputs, 1):
                f.write(f"TRIAL {i}\n")
                f.write("-"*80 + "\n")
                f.write(output)
                f.write("\n\n" + "="*80 + "\n\n")
        print(f"\nOutput written to: {output_file}")
    else:
        # Single trial - save to base file
        output_file = os.path.join(RESULTS_DIR, f"{output_base}.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(outputs[0])
        print(f"Output written to: {output_file}")
    
    return outputs

