#!/usr/bin/env python3
"""
Script to run a prompt through a GPT model and save the output to a file.
"""

import argparse
import os
from openai import OpenAI
from datetime import datetime

# Default folders
PROMPTS_DIR = "prompts"
RESULTS_DIR = "results"


def ensure_directories():
    """Create prompts/ and results/ directories if they don't exist."""
    os.makedirs(PROMPTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def read_prompt(prompt_input):
    """Read prompt from file or use as direct text.
    
    First checks if prompt_input is a direct file path.
    If not found and doesn't already start with prompts/, checks in prompts/ folder.
    If still not found, treats as direct text.
    """
    # Check if it's an absolute path or exists in current directory
    if os.path.isfile(prompt_input):
        with open(prompt_input, 'r', encoding='utf-8') as f:
            return f.read()
    
    # Only check in prompts/ folder if it doesn't already start with prompts/
    if not (prompt_input.startswith(PROMPTS_DIR + os.sep) or prompt_input.startswith(PROMPTS_DIR + '/')):
        prompts_path = os.path.join(PROMPTS_DIR, prompt_input)
        if os.path.isfile(prompts_path):
            with open(prompts_path, 'r', encoding='utf-8') as f:
                return f.read()
    
    # If not found as file, treat as direct text
    return prompt_input


def call_gpt(prompt, model, temperature=0.7):
    """Call GPT model with the given prompt."""
    client = OpenAI()
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling GPT API: {e}")
        raise


def write_output(output, output_file):
    """Write output to file."""
    # Ensure results directory exists
    ensure_directories()
    
    # If output_file is not already in results/, put it there
    if not output_file.startswith(RESULTS_DIR) and not os.path.isabs(output_file):
        output_file = os.path.join(RESULTS_DIR, output_file)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output)
    print(f"Output written to: {output_file}")


def run_trials(prompt_text, model, temperature, num_trials, output_base):
    """Run multiple trials and save all outputs."""
    outputs = []
    
    print(f"Running {num_trials} trials...")
    
    for trial in range(1, num_trials + 1):
        print(f"\nTrial {trial}/{num_trials}...")
        try:
            output = call_gpt(prompt_text, model, temperature)
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
        write_output(outputs[0], f"{output_base}.txt")
    
    return outputs


def main():
    parser = argparse.ArgumentParser(
        description='Run a prompt through a GPT model and save output to a file'
    )
    parser.add_argument(
        'prompt',
        help='Path to prompt file (searches in prompts/ folder if not found) or prompt text directly'
    )
    parser.add_argument(
        '--model',
        '-m',
        default='gpt-4o',
        help='GPT model to use (default: gpt-4o)'
    )
    parser.add_argument(
        '--output',
        '-o',
        help='Output file path (default: auto-generated based on timestamp)'
    )
    parser.add_argument(
        '--temperature',
        '-t',
        type=float,
        default=0.7,
        help='Temperature for generation (default: 0.7)'
    )
    parser.add_argument(
        '--trials',
        '-n',
        type=int,
        default=1,
        help='Number of trials to run (default: 1)'
    )
    
    args = parser.parse_args()
    
    # Ensure directories exist
    ensure_directories()
    
    # Read prompt
    print(f"Reading prompt from: {args.prompt}")
    prompt_text = read_prompt(args.prompt)
    
    # Determine the base name for output file
    if args.output:
        output_base = os.path.splitext(args.output)[0]  # Remove extension if present
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Try to get base name from prompt file
        if os.path.isfile(args.prompt):
            base_name = os.path.splitext(os.path.basename(args.prompt))[0]
        elif os.path.isfile(os.path.join(PROMPTS_DIR, args.prompt)):
            base_name = os.path.splitext(args.prompt)[0]
        else:
            base_name = "prompt"
        output_base = f"{base_name}_output_{timestamp}"
    
    # Run trials
    print(f"Using model: {args.model}")
    outputs = run_trials(prompt_text, args.model, args.temperature, args.trials, output_base)
    
    # Print summary to console
    print("\n" + "="*80)
    if args.trials == 1:
        print("OUTPUT:")
        print("="*80)
        print(outputs[0])
    else:
        print(f"SUMMARY: {args.trials} trials completed")
        print("="*80)
        for i, output in enumerate(outputs, 1):
            print(f"\nTrial {i}:")
            print("-"*80)
            print(output[:500] + "..." if len(output) > 500 else output)


if __name__ == "__main__":
    main()