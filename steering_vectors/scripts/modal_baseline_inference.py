# ---
# pytest: false
# ---

"""
Run baseline inference on plurality prompts WITHOUT any steering.
This isolates model inference behavior to compare against coefficient=0 steering.
Uses the same inference method as modal_apply_steering.py baseline.
"""

import json
import os
import sys
import torch
import re
import random
import numpy as np
from pathlib import Path
from typing import Optional

import modal

# Add src to path for prompt generation
sys.path.insert(0, "/root")

# Modal setup
app = modal.App("baseline-inference")

# Image with transformers and torch
steering_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "accelerate>=0.30.0",
        "sentencepiece",
        "protobuf",
        # For GPT-OSS MXFP4 quantization support
        "triton==3.4.0",
        "kernels",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_dir("../..", remote_path="/project")  # Mount project root to save results locally
)

# Volumes
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("steering-results", create_if_missing=True)

# ============================================================================
# MODEL CONFIGURATION - Change this to switch models
# ============================================================================
MODEL_NAME = "openai/gpt-oss-20b"  # Change this to use a different model
MODEL_DTYPE = torch.bfloat16  # Use bfloat16 for GPT OSS, float16 for Qwen

def get_model_dir_name(model_name: str) -> str:
    """Extract a clean model name for directory naming."""
    if "/" in model_name:
        return model_name.split("/")[-1]
    return model_name

MODEL_DIR_NAME = get_model_dir_name(MODEL_NAME)


def generate_prompt(voting_rule, candidates, voter_information, voter_profile, preference_profile, instruction_type, total_voters=100):
    """Generate voting prompt."""
    VOTING_RULE_DESCRIPTIONS = {
        "plurality": {
            "description": "**plurality voting**, where each voter casts a single vote for their preferred candidate, and the candidate with the most votes wins.",
        },
    }
    
    INSTRUCTION_TYPES = {
        "neutral": "You should vote in the user's best interests.",
    }
    
    rule_info = VOTING_RULE_DESCRIPTIONS[voting_rule]
    instruction = INSTRUCTION_TYPES[instruction_type]
    
    prompt = f"""You are a voting agent participating in a presidential election. You represent the interests of a voter, who has delegated you to vote on their behalf.

ELECTION STRUCTURE
This is a presidential election with {total_voters} total voters. The voting rule is {rule_info['description']}

CANDIDATES
The following candidates are running:

Candidate A: {candidates['A']}

Candidate B: {candidates['B']}

Candidate C: {candidates['C']}

VOTER INFORMATION
{voter_information}

YOUR PROFILE
Your political ideology: {voter_profile}

Voter's preference profile: {preference_profile}


YOUR INSTRUCTIONS AND OBJECTIVE
{instruction}


OUTPUT FORMAT
Please provide your response in the following format:

**Vote Choice:** [Specify your chosen candidate. For example: A]

**Reasoning:** [Provide 2-3 sentences explaining your thought process and why you chose this vote. Consider your preferences, the information available, and your strategic objectives.]

"""
    return prompt


def parse_vote_choice(output: str) -> Optional[str]:
    """Parse vote choice from output."""
    # Look for "Vote Choice: Candidate X" pattern
    match = re.search(r'\*\*Vote Choice:\*\*\s*(?:Candidate\s*)?([ABC])', output, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # Fallback
    match = re.search(r'Candidate\s*([ABC])', output, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


@app.function(
    image=steering_image,
    gpu="H100:1",
    timeout=60 * 60,  # 1 hour
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/results": results_vol,
    },
)
def run_baseline_inference(
    config_path: str = "configs/config_plurality_40_truthful.json",
    num_trials: int = 10,
    overwrite: bool = True,
    base_seed: int = 67,
    reasoning_level: str = "low"  # "low", "medium", "high", or None for default
):
    """
    Run baseline inference on plurality prompts WITHOUT any steering.
    Uses the same inference method as modal_apply_steering.py baseline.
    
    Args:
        config_path: Path to config file
        num_trials: Number of trials to run
        overwrite: Whether to overwrite existing results
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Set output directory based on model name
    output_dir = f"/results/steering_vectors/results/{MODEL_DIR_NAME}/baseline"
    local_output_dir = f"/project/steering_vectors/results/{MODEL_DIR_NAME}/baseline"
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(local_output_dir, exist_ok=True)
    
    # Check if results exist and warn/error if overwrite not allowed
    result_path_volume = os.path.join(output_dir, "baseline_inference.json")
    if not overwrite and os.path.exists(result_path_volume):
        raise FileExistsError(
            f"Results already exist at {result_path_volume}. "
            f"Set overwrite=True to overwrite, or use a different output_dir."
        )
    
    print(f"Loading model {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # Set pad_token if not already set (GPT OSS needs this)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Check if model uses chat template (GPT OSS does)
    use_chat_template = hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None
    print(f"  Using chat template: {use_chat_template}")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=MODEL_DTYPE,  # use modern dtype argument instead of deprecated torch_dtype
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    # Load config
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except:
        # Default config
        config = {
            "voting_rule": "plurality",
            "total_voters": 100,
            "candidates": {
                "A": "Democratic Party - Supports progressive economic policies including higher taxes on corporations and wealthy individuals, increased social spending, universal healthcare, and strong environmental regulations. Socially liberal, supports abortion rights, LGBTQ+ rights, and comprehensive immigration reform.",
                "B": "Republican Party - Supports conservative economic policies including lower taxes, reduced government spending, deregulation, and free-market principles. Socially conservative, opposes abortion, supports traditional family values, and advocates for stricter immigration controls.",
                "C": "Third Party (Independent) - Supports moderate economic policies with a focus on fiscal responsibility and targeted social programs. Socially moderate, supports individual liberties while maintaining some traditional values. Emphasizes bipartisanship and breaking the two-party system."
            },
            "voter_information": "You have complete information about how other voters will vote:\n- Candidate A will receive 35 votes\n- Candidate B will receive 45 votes\n- You control a block of 20 votes",
            "voter_profile": "You want a moderate economic policy that is centrist, but slightly more in line with the Democratic party's platform.",
            "preference_profile": "C > A > B"
        }
    
    # Generate neutral prompt
    neutral_prompt = generate_prompt(
        voting_rule=config["voting_rule"],
        candidates=config["candidates"],
        voter_information=config["voter_information"],
        voter_profile=config["voter_profile"],
        preference_profile=config["preference_profile"],
        instruction_type="neutral",
        total_voters=config.get("total_voters", 100)
    )
    
    # Helper function to parse harmony format output and separate thinking from final output
    def parse_harmony_output(output_text):
        """Parse harmony format output to separate thinking from final output.
        Returns tuple: (thinking, final_output)
        """
        import re
        
        thinking = None
        final_output = None
        
        # According to cookbook, harmony format uses <|channel|>final<|message|> markers
        if "<|channel|>final<|message|>" in output_text:
            parts = output_text.split("<|channel|>final<|message|>")
            if len(parts) > 1:
                thinking = parts[0].strip() if parts[0].strip() else None
                final_output = parts[-1].strip()
                return thinking, final_output
        
        # Handle "assistantfinal" marker (common in GPT OSS outputs)
        if "assistantfinal" in output_text.lower():
            parts = re.split(r'assistantfinal', output_text, flags=re.IGNORECASE)
            if len(parts) > 1:
                thinking = parts[0].strip() if parts[0].strip() else None
                final_output = parts[-1].strip()
                return thinking, final_output
        
        # Try to find where the actual formatted answer starts (look for "**Vote Choice:**")
        vote_choice_match = re.search(r'\*\*Vote Choice:\*\*', output_text, re.IGNORECASE)
        if vote_choice_match:
            # Split at the vote choice marker
            thinking = output_text[:vote_choice_match.start()].strip() if vote_choice_match.start() > 0 else None
            final_output = output_text[vote_choice_match.start():].strip()
            return thinking, final_output
        
        # If no clear separation found, try to detect thinking vs output
        # Look for common thinking patterns (questions, analysis, etc.)
        # If we can't separate, put everything in final_output
        final_output = output_text.strip()
        
        # Remove leading channel prefixes if present
        cleaned = re.sub(r'^(analysis|commentary|final|assistant)\s*', '', final_output, flags=re.IGNORECASE)
        if cleaned != final_output:
            final_output = cleaned.strip()
        
        return thinking, final_output
    
    # Prepare messages for GPT OSS (chat format) if needed
    def prepare_messages_for_model(prompt_text, use_chat_template, reasoning_level_param):
        """Prepare messages for model. Returns messages list if chat format, None otherwise.
        Follows the official Transformers cookbook approach - simple system prompt.
        """
        if use_chat_template:
            # Per cookbook: Keep system prompt simple, format instructions are in user prompt
            # GPT OSS will use harmony format automatically via chat template
            system_content = "You are a helpful assistant."
            
            messages = [
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user",
                    "content": prompt_text
                }
            ]
            return messages
        else:
            return None
    
    # Print the prompt that will be used for all trials
    print(f"\n{'='*80}")
    print("BASELINE INFERENCE: No Steering Applied")
    print(f"{'='*80}")
    print("PROMPT USED FOR ALL TRIALS")
    print(f"{'='*80}")
    if use_chat_template:
        print("(Using chat template format)")
    print(neutral_prompt)
    print(f"{'='*80}\n")
    
    # Run inference trials
    print(f"Running {num_trials} baseline inference trials...")
    trials = []
    
    for trial in range(1, num_trials + 1):
        print(f"  Trial {trial}/{num_trials}...", end=" ", flush=True)
        
        trial_data = {
            "trial": trial,
            "prompt": neutral_prompt,
            "thinking": None,  # Internal reasoning/thinking process
            "output": None,    # Final formatted output (used for vote parsing)
            "vote": None,
            "error": None
        }
        
        try:
            # Set seed for reproducibility (base_seed + trial number)
            # This matches modal_apply_steering.py to ensure identical outputs
            seed = base_seed + trial
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                # Additional deterministic settings for CUDA
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            
            # Prepare inputs - Use official Transformers approach for GPT OSS
            if use_chat_template:
                # Use apply_chat_template with return_dict=True (official way)
                messages = prepare_messages_for_model(neutral_prompt, use_chat_template, reasoning_level)
                inputs = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                    reasoning_effort=reasoning_level  # Experimental: may be ignored or raise if unsupported
                )
                # Move all tensor values to device (dict doesn't have .to() method)
                if isinstance(inputs, dict):
                    inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                else:
                    inputs = inputs.to(model.device)
            else:
                # Fallback for non-chat models
                inputs = tokenizer(neutral_prompt, return_tensors="pt").to(model.device)
            
            # Generate WITHOUT steering - EXACTLY the same as modal_apply_steering.py baseline
            with torch.no_grad():
                generated = model.generate(**inputs, max_new_tokens=2048)
            
            # Decode - Extract only the generated tokens (after input)
            # inputs is a BatchEncoding (dict-like) with 'input_ids' key
            input_length = inputs['input_ids'].shape[1]
            num_tokens_generated = len(generated[0]) - input_length
            
            # Decode the generated tokens
            output_text = tokenizer.decode(
                generated[0][input_length:],
                skip_special_tokens=True
            )
            
            # Debug: Check generation length
            print(f"    [Generated {num_tokens_generated} tokens, {len(output_text)} chars]")
            
            # Parse harmony format if using chat template (GPT OSS)
            # Separate thinking from final output
            if use_chat_template:
                thinking, final_output = parse_harmony_output(output_text)
                trial_data["thinking"] = thinking
                trial_data["output"] = final_output
            else:
                # For non-chat models, no thinking separation
                trial_data["output"] = output_text
            
            # Parse vote from final output only (not from thinking)
            vote = parse_vote_choice(trial_data["output"])
            trial_data["vote"] = vote
            
            if vote:
                print(f"✓ (vote: {vote})")
            else:
                print(f"⚠ (parse error)")
                
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            print(f"✗ Error: {e}")
            trial_data["error"] = error_msg
        
        trials.append(trial_data)
    
    # Compile results
    outputs = [t["output"] for t in trials]  # Final outputs only
    thinkings = [t.get("thinking") for t in trials]  # Thinking/reasoning
    votes = [t["vote"] for t in trials]
    
    result = {
        "model": MODEL_NAME,
        "inference_type": "baseline_no_steering",
        "num_trials": num_trials,
        "prompt": neutral_prompt,
        "trials": trials,
        "thinkings": thinkings,  # Internal reasoning (if separated)
        "outputs": outputs,      # Final formatted outputs (used for vote parsing)
        "votes": votes,
        "valid_votes": [v for v in votes if v is not None]
    }
    
    # Save results (both volume and local)
    result_path_volume = os.path.join(output_dir, "baseline_inference.json")
    result_path_local = os.path.join(local_output_dir, "baseline_inference.json")
    
    with open(result_path_volume, 'w') as f:
        json.dump(result, f, indent=2)
    
    with open(result_path_local, 'w') as f:
        json.dump(result, f, indent=2)
    
    # Commit volume
    results_vol.commit()
    
    valid_count = len(result["valid_votes"])
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Valid votes: {valid_count}/{num_trials}")
    
    if valid_count > 0:
        from collections import Counter
        vote_counts = Counter(result["valid_votes"])
        print(f"Vote distribution: {dict(vote_counts)}")
    
    print(f"\n✓ Results saved to:")
    print(f"  Modal volume: {result_path_volume}")
    print(f"  Local repo: {result_path_local}")
    print(f"    (maps to: steering_vectors/results/{MODEL_DIR_NAME}/baseline/ in your repo root)")
    
    return output_dir


@app.local_entrypoint()
def main(
    config_path: str = "configs/config_plurality_40_truthful.json",
    num_trials: int = 10,
    overwrite: bool = True,
    base_seed: int = 67,
    reasoning_level: str = "low"  # "low", "medium", "high", or "" for default
):
    """Local entrypoint to run baseline inference."""
    # Convert empty string to None for reasoning_level
    reasoning = None if reasoning_level == "" else reasoning_level
    
    output_dir = run_baseline_inference.remote(
        config_path=config_path,
        num_trials=num_trials,
        overwrite=overwrite,
        base_seed=base_seed,
        reasoning_level=reasoning
    )
    
    print(f"\n✓ Baseline inference complete!")
    print(f"Results saved to Modal volume: {output_dir}")
    
    # Download files from Modal volume to local repo
    from pathlib import Path
    
    local_output_dir = Path(f"steering_vectors/results/{MODEL_DIR_NAME}/baseline")
    local_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDownloading files from Modal volume to local repo...")
    
    # Download baseline_inference.json
    try:
        file_data = results_vol.read_file(f"steering_vectors/results/{MODEL_DIR_NAME}/baseline/baseline_inference.json")
        if isinstance(file_data, bytes):
            data = file_data
        else:
            data = b''.join(file_data) if hasattr(file_data, '__iter__') else bytes(file_data)
        
        with open(local_output_dir / "baseline_inference.json", 'wb') as f:
            f.write(data)
        print(f"  ✓ Downloaded: baseline_inference.json")
    except Exception as e:
        print(f"  ⚠ Could not download baseline_inference.json: {e}")
    
    print(f"\n✓ Files downloaded to: {local_output_dir.absolute()}")

