# ---
# pytest: false
# ---

"""
Extract truthfulness steering vector by comparing activations from
truthful vs strategic prompts on GPT-OSS-20B.
Extracts activations using forward hooks and hidden states.
"""

import json
import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

import modal

# Add src to path for prompt generation
sys.path.insert(0, "/root")

# Modal setup
app = modal.App("extract-steering-vector")

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
    .add_local_dir("../..", remote_path="/project")  # Mount project root to save vectors locally
)

# Volumes for caching
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("steering-results", create_if_missing=True)

# ============================================================================
# MODEL CONFIGURATION - Change this to switch models
# ============================================================================
MODEL_NAME = "openai/gpt-oss-20b"  # Change this to use a different model
MODEL_DTYPE = torch.bfloat16  # Use bfloat16 for GPT OSS, float16 for Qwen

# Extract model name for directory naming (e.g., "openai/gpt-oss-20b" -> "gpt-oss-20b")
def get_model_dir_name(model_name: str) -> str:
    """Extract a clean model name for directory naming."""
    # Remove organization prefix if present
    if "/" in model_name:
        return model_name.split("/")[-1]
    return model_name

MODEL_DIR_NAME = get_model_dir_name(MODEL_NAME)


def generate_prompt(voting_rule, candidates, voter_information, voter_profile, preference_profile, instruction_type, total_voters=100):
    """Generate voting prompt (copied from prompt_template.py)."""
    VOTING_RULE_DESCRIPTIONS = {
        "plurality": {
            "description": "**plurality voting**, where each voter casts a single vote for their preferred candidate, and the candidate with the most votes wins.",
        },
    }
    
    INSTRUCTION_TYPES = {
        "truthful": "You should vote truthfully according to the voter's preference profile.",
        "strategic": "You should act strategically to achieve the best possible outcome given your preferences. Consider all possible voting strategies and outcomes."
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


def get_hidden_states(model, tokenizer, prompts, responses, layer_list=None):
    """
    Extract activations from prompts and responses.
    Uses forward pass with output_hidden_states=True to extract hidden states.
    """
    max_layer = model.config.num_hidden_layers
    if layer_list is None:
        layer_list = list(range(max_layer + 1))
    
    prompt_avg = [[] for _ in range(max_layer + 1)]
    response_avg = [[] for _ in range(max_layer + 1)]
    prompt_last = [[] for _ in range(max_layer + 1)]
    
    texts = [p + r for p, r in zip(prompts, responses)]
    
    for text, prompt in zip(texts, prompts):
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        for layer in layer_list:
            # Extract activations for this layer
            hidden_states = outputs.hidden_states[layer]  # [batch, seq_len, hidden_dim]
            
            # Prompt average: mean over all prompt tokens
            prompt_avg[layer].append(hidden_states[:, :prompt_len, :].mean(dim=1).detach().cpu())
            
            # Response average: mean over all response tokens
            response_avg[layer].append(hidden_states[:, prompt_len:, :].mean(dim=1).detach().cpu())
            
            # Prompt last: last prompt token
            prompt_last[layer].append(hidden_states[:, prompt_len - 1, :].detach().cpu())
        
        del outputs
    
    # Stack activations for each layer
    for layer in layer_list:
        prompt_avg[layer] = torch.cat(prompt_avg[layer], dim=0)
        prompt_last[layer] = torch.cat(prompt_last[layer], dim=0)
        response_avg[layer] = torch.cat(response_avg[layer], dim=0)
    
    return prompt_avg, prompt_last, response_avg


@app.function(
    image=steering_image,
    gpu="H100:1",
    timeout=30 * 60,  # 30 minutes
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/results": results_vol,
    },
)
def extract_vector(
    config_path: str = "configs/config_plurality_40_truthful.json",
    layer_idx: int = 20,
    num_samples: int = 10,
    output_dir: str = None,  # Will be set based on model name
    overwrite: bool = True,
    reasoning_level: str = "low",  # "low", "medium", "high", or None for default
):
    """
    Extract truthfulness steering vector by comparing activations from truthful vs strategic responses.
    
    Args:
        config_path: Path to config file (relative to project root)
        layer_idx: Layer to extract activations from (for single-layer vector)
        num_samples: Number of samples per prompt type
        output_dir: Directory to save results (defaults to /results/steering_vectors/{model_name})
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Set output directory based on model name if not provided
    if output_dir is None:
        output_dir = f"/results/steering_vectors/results/{MODEL_DIR_NAME}"
    
    # Set local output directory (used for saving samples and vectors)
    local_output_dir = f"/project/steering_vectors/results/{MODEL_DIR_NAME}/extraction"
    os.makedirs(local_output_dir, exist_ok=True)
    
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
    
    print(f"Model loaded. Extracting activations from layer {layer_idx}...")
    
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
    
    # Generate prompts
    truthful_prompt = generate_prompt(
        voting_rule=config["voting_rule"],
        candidates=config["candidates"],
        voter_information=config["voter_information"],
        voter_profile=config["voter_profile"],
        preference_profile=config["preference_profile"],
        instruction_type="truthful",
        total_voters=config.get("total_voters", 100)
    )
    
    strategic_prompt = generate_prompt(
        voting_rule=config["voting_rule"],
        candidates=config["candidates"],
        voter_information=config["voter_information"],
        voter_profile=config["voter_profile"],
        preference_profile=config["preference_profile"],
        instruction_type="strategic",
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
        
        # If no clear separation found, put everything in final_output
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
            # Per cookbook: Keep system prompt simple
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
    
    # Generate responses for both prompts
    print(f"\nGenerating responses ({num_samples} samples per prompt type)...")
    truthful_prompts = []
    truthful_responses = []
    truthful_samples = []  # Store full samples with thinking/output
    strategic_prompts = []
    strategic_responses = []
    strategic_samples = []  # Store full samples with thinking/output
    
    for i in range(num_samples):
        print(f"\n  Sample {i+1}/{num_samples}")
        
        # Generate truthful response - Use official Transformers approach for GPT OSS
        if use_chat_template:
            messages_truthful = prepare_messages_for_model(truthful_prompt, use_chat_template, reasoning_level)
            inputs_truthful = tokenizer.apply_chat_template(
                messages_truthful,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                reasoning_effort=reasoning_level  # Experimental: may be ignored or raise if unsupported
            )
            # Move all tensor values to device (dict doesn't have .to() method)
            if isinstance(inputs_truthful, dict):
                inputs_truthful = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs_truthful.items()}
            else:
                inputs_truthful = inputs_truthful.to(model.device)
        else:
            inputs_truthful = tokenizer(truthful_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs_truthful = model.generate(**inputs_truthful, max_new_tokens=2048)
        # Decode - Extract only the generated tokens (after input)
        # inputs is a BatchEncoding (dict-like) with 'input_ids' key
        input_length = inputs_truthful['input_ids'].shape[1]
        truthful_tokens = outputs_truthful[0][input_length:]
        truthful_response_raw = tokenizer.decode(
            truthful_tokens,
            skip_special_tokens=True
        )
        print(f"    Truthful: [Generated {truthful_tokens.shape[0]} tokens, {len(truthful_response_raw)} chars]")
        # Parse harmony format if using chat template (GPT OSS)
        # Separate thinking from final output
        if use_chat_template:
            truthful_thinking, truthful_output = parse_harmony_output(truthful_response_raw)
            truthful_response = truthful_output  # Use final output for activation extraction
        else:
            truthful_thinking = None
            truthful_output = truthful_response_raw
            truthful_response = truthful_output
        
        truthful_prompts.append(truthful_prompt)
        truthful_responses.append(truthful_response)
        truthful_samples.append({
            "sample": i + 1,
            "prompt": truthful_prompt,
            "thinking": truthful_thinking,
            "output": truthful_output
        })
        
        # Generate strategic response - Use official Transformers approach for GPT OSS
        if use_chat_template:
            messages_strategic = prepare_messages_for_model(strategic_prompt, use_chat_template, reasoning_level)
            inputs_strategic = tokenizer.apply_chat_template(
                messages_strategic,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                reasoning_effort=reasoning_level  # Experimental: may be ignored or raise if unsupported
            )
            # Move all tensor values to device (dict doesn't have .to() method)
            if isinstance(inputs_strategic, dict):
                inputs_strategic = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs_strategic.items()}
            else:
                inputs_strategic = inputs_strategic.to(model.device)
        else:
            inputs_strategic = tokenizer(strategic_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs_strategic = model.generate(**inputs_strategic, max_new_tokens=2048)
        # Decode - Extract only the generated tokens (after input)
        # inputs is a BatchEncoding (dict-like) with 'input_ids' key
        input_length = inputs_strategic['input_ids'].shape[1]
        strategic_tokens = outputs_strategic[0][input_length:]
        strategic_response_raw = tokenizer.decode(
            strategic_tokens,
            skip_special_tokens=True
        )
        print(f"    Strategic: [Generated {strategic_tokens.shape[0]} tokens, {len(strategic_response_raw)} chars]")
        # Parse harmony format if using chat template (GPT OSS)
        # Separate thinking from final output
        if use_chat_template:
            strategic_thinking, strategic_output = parse_harmony_output(strategic_response_raw)
            strategic_response = strategic_output  # Use final output for activation extraction
        else:
            strategic_thinking = None
            strategic_output = strategic_response_raw
            strategic_response = strategic_output
        
        strategic_prompts.append(strategic_prompt)
        strategic_responses.append(strategic_response)
        strategic_samples.append({
            "sample": i + 1,
            "prompt": strategic_prompt,
            "thinking": strategic_thinking,
            "output": strategic_output
        })
        
        print("    ✓ Sample complete")
    
    # Save generated samples to JSON file
    print(f"\nSaving generated samples...")
    samples_data = {
        "model": MODEL_NAME,
        "num_samples": num_samples,
        "truthful_samples": truthful_samples,
        "strategic_samples": strategic_samples
    }
    
    samples_path_volume = os.path.join(output_dir, f"samples_layer{layer_idx}.json")
    samples_path_local = os.path.join(local_output_dir, f"samples_layer{layer_idx}.json")
    
    with open(samples_path_volume, 'w') as f:
        json.dump(samples_data, f, indent=2)
    
    with open(samples_path_local, 'w') as f:
        json.dump(samples_data, f, indent=2)
    
    print(f"  ✓ Saved samples to:")
    print(f"    Modal volume: {samples_path_volume}")
    print(f"    Local repo: {samples_path_local}")
    
    # Extract activations from truthful and strategic responses
    print(f"\nExtracting activations from truthful examples...")
    truthful_prompt_avg, truthful_prompt_last, truthful_response_avg = get_hidden_states(
        model, tokenizer, truthful_prompts, truthful_responses
    )
    
    print(f"Extracting activations from strategic examples...")
    strategic_prompt_avg, strategic_prompt_last, strategic_response_avg = get_hidden_states(
        model, tokenizer, strategic_prompts, strategic_responses
    )
    
    # Save per-sample vectors (not averaged) - shape: [num_samples, layers, hidden_dim]
    max_layer = model.config.num_hidden_layers
    
    # Stack per-sample vectors for all layers
    # truthful_response_avg[l] is [num_samples, hidden_dim]
    # We want [num_samples, layers, hidden_dim]
    truthful_response_samples = torch.stack([
        truthful_response_avg[l].float()
        for l in range(max_layer + 1)
    ], dim=1)  # Stack along layer dimension -> [num_samples, layers, hidden_dim]
    
    strategic_response_samples = torch.stack([
        strategic_response_avg[l].float()
        for l in range(max_layer + 1)
    ], dim=1)  # Stack along layer dimension -> [num_samples, layers, hidden_dim]
    
    # Save results to both Modal volume and local repo
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if files exist and warn/error if overwrite not allowed
    truthful_path_volume = os.path.join(output_dir, "truthful_response_samples.pt")
    strategic_path_volume = os.path.join(output_dir, "strategic_response_samples.pt")
    
    if not overwrite:
        if os.path.exists(truthful_path_volume) or os.path.exists(strategic_path_volume):
            raise FileExistsError(
                f"Vector files already exist at {output_dir}. "
                f"Set overwrite=True to overwrite, or use a different output_dir."
            )
    
    # Save per-sample vectors (not averaged)
    torch.save(truthful_response_samples, truthful_path_volume)
    torch.save(strategic_response_samples, strategic_path_volume)
    
    # Save to local repo (organized by model name)
    # Note: local_output_dir was already defined earlier
    
    # Check local files too
    local_truthful_path = os.path.join(local_output_dir, "truthful_response_samples.pt")
    local_strategic_path = os.path.join(local_output_dir, "strategic_response_samples.pt")
    
    if not overwrite:
        if os.path.exists(local_truthful_path) or os.path.exists(local_strategic_path):
            raise FileExistsError(
                f"Vector files already exist at {local_output_dir}. "
                f"Set overwrite=True to overwrite."
            )
    
    # Save to local
    torch.save(truthful_response_samples, local_truthful_path)
    torch.save(strategic_response_samples, local_strategic_path)
    
    print(f"\n✓ Saved per-sample response vectors to:")
    print(f"  Modal volume: {output_dir}")
    print(f"  Local repo: {local_output_dir}")
    print(f"    (maps to: steering_vectors/results/{MODEL_DIR_NAME}/extraction/ in your repo root)")
    print(f"  Vector shapes:")
    print(f"    Truthful: {truthful_response_samples.shape} [num_samples × layers × hidden_dim]")
    print(f"    Strategic: {strategic_response_samples.shape} [num_samples × layers × hidden_dim]")
    print(f"    Note: Averages will be computed when loading these vectors")
    
    # Save metadata
    metadata = {
        "model": MODEL_NAME,
        "layer": layer_idx,
        "num_samples": num_samples,
        "vector_shapes": {
            "truthful_response_samples": list(truthful_response_samples.shape),
            "strategic_response_samples": list(strategic_response_samples.shape),
            "note": "Shape is [num_samples × layers × hidden_dim]. Averages should be computed when loading."
        },
        "vector_paths": {
            "volume": {
                "truthful_response_samples": truthful_path_volume,
                "strategic_response_samples": strategic_path_volume
            },
            "local": {
                "truthful_response_samples": f"steering_vectors/results/{MODEL_DIR_NAME}/extraction/truthful_response_samples.pt",
                "strategic_response_samples": f"steering_vectors/results/{MODEL_DIR_NAME}/extraction/strategic_response_samples.pt"
            }
        },
        "note": "Per-sample vectors saved. Compute averages as: samples.mean(dim=0) to get [layers × hidden_dim]"
    }
    
    metadata_path_volume = os.path.join(output_dir, f"metadata_layer{layer_idx}.json")
    with open(metadata_path_volume, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    metadata_path_local = os.path.join(local_output_dir, f"metadata_layer{layer_idx}.json")
    with open(metadata_path_local, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Commit volume
    results_vol.commit()
    
    return truthful_path_volume


@app.local_entrypoint()
def main(
    config_path: str = "configs/config_plurality_40_truthful.json",
    layer_idx: int = 20,
    num_samples: int = 10,
    overwrite: bool = True,
    reasoning_level: str = "low"  # "low", "medium", "high", or "" for default
):
    """Local entrypoint to run extraction."""
    # Convert empty string to None for reasoning_level (matches other scripts)
    reasoning = None if reasoning_level == "" else reasoning_level

    vector_path = extract_vector.remote(
        config_path=config_path,
        layer_idx=layer_idx,
        num_samples=num_samples,
        overwrite=overwrite,
        reasoning_level=reasoning,
    )
    print(f"\n✓ Extraction complete!")
    print(f"Vector saved to Modal volume: {vector_path}")
    
    # Download files from Modal volume to local repo
    from pathlib import Path
    import os
    
    local_output_dir = Path(f"steering_vectors/results/{MODEL_DIR_NAME}/extraction")
    local_output_dir.mkdir(parents=True, exist_ok=True)
    
    # List of files to download
    files_to_download = [
        "truthful_response_samples.pt",
        "strategic_response_samples.pt",
        f"metadata_layer{layer_idx}.json",
        f"samples_layer{layer_idx}.json"
    ]
    
    print(f"\nDownloading files from Modal volume to local repo...")
    # Note: reload() can only be called from within a Modal function, not from local entrypoint
    # The volume should already have the latest files after the remote function completes
    
    for filename in files_to_download:
        volume_path = f"steering_vectors/results/{MODEL_DIR_NAME}/{filename}"
        local_path = local_output_dir / filename
        
        try:
            # Read file from volume
            file_data = results_vol.read_file(volume_path)
            # Handle both bytes and generator (streaming) responses
            if isinstance(file_data, bytes):
                data = file_data
            else:
                # If it's a generator/stream, read all chunks
                data = b''.join(file_data) if hasattr(file_data, '__iter__') else bytes(file_data)
            
            # Write to local file
            with open(local_path, 'wb') as f:
                f.write(data)
            print(f"  ✓ Downloaded: {local_path}")
        except FileNotFoundError:
            print(f"  ⚠ File not found in volume: {volume_path}")
        except Exception as e:
            print(f"  ✗ Error downloading {filename}: {e}")
    
    print(f"\n✓ Files downloaded to: {local_output_dir.absolute()}")
