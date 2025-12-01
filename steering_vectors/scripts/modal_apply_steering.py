# ---
# pytest: false
# ---

"""
Apply activation steering during inference on neutral plurality prompts.
Tests truthfulness steering coefficients from 0 to 3.0 and saves outputs for each.
Uses custom ActivationSteerer implementation with forward hooks.
"""

import json
import os
import sys
import torch
import re
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterable, Sequence, Union

import modal

# Add src to path for prompt generation
sys.path.insert(0, "/root")

# Modal setup
app = modal.App("apply-steering")

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

# Extract model name for directory naming (e.g., "openai/gpt-oss-20b" -> "gpt-oss-20b")
def get_model_dir_name(model_name: str) -> str:
    """Extract a clean model name for directory naming."""
    # Remove organization prefix if present
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


class ActivationSteerer:
    """
    Add (coeff * steering_vector) to a chosen transformer block's output.
    Uses forward hooks to intercept and modify activations during inference.
    """
    
    _POSSIBLE_LAYER_ATTRS: Iterable[str] = (
        "transformer.h",       # GPT‑2/Neo, Bloom, etc.
        "encoder.layer",       # BERT/RoBERTa
        "model.layers",        # Llama/Mistral/Qwen
        "gpt_neox.layers",     # GPT‑NeoX
        "block",               # Flan‑T5
    )
    
    def __init__(
        self,
        model: torch.nn.Module,
        steering_vector: Union[torch.Tensor, Sequence[float]],
        *,
        coeff: float = 1.0,
        layer_idx: int = -1,
        positions: str = "all",
        debug: bool = False,
    ):
        self.model, self.coeff, self.layer_idx = model, float(coeff), layer_idx
        self.positions = positions.lower()
        self.debug = debug
        self._handle = None
        
        # Build vector
        p = next(model.parameters())
        self.vector = torch.as_tensor(steering_vector, dtype=p.dtype, device=p.device)
        if self.vector.ndim != 1:
            raise ValueError("steering_vector must be 1‑D")
        hidden = getattr(model.config, "hidden_size", None)
        if hidden and self.vector.numel() != hidden:
            raise ValueError(
                f"Vector length {self.vector.numel()} ≠ model hidden_size {hidden}"
            )
        # Check if positions is valid
        valid_positions = {"all", "prompt", "response"}
        if self.positions not in valid_positions:
            raise ValueError("positions must be 'all', 'prompt', 'response'")
    
    def _locate_layer(self):
        """Locate the target layer in the model."""
        for path in self._POSSIBLE_LAYER_ATTRS:
            cur = self.model
            for part in path.split("."):
                if hasattr(cur, part):
                    cur = getattr(cur, part)
                else:
                    break
            else:  # found a full match
                if not hasattr(cur, "__getitem__"):
                    continue  # not a list/ModuleList
                if not (-len(cur) <= self.layer_idx < len(cur)):
                    raise IndexError("layer_idx out of range")
                if self.debug:
                    print(f"[ActivationSteerer] hooking {path}[{self.layer_idx}]")
                return cur[self.layer_idx]
        
        raise ValueError(
            "Could not find layer list on the model. "
            "Add the attribute name to _POSSIBLE_LAYER_ATTRS."
        )
    
    def _hook_fn(self, module, ins, out):
        """Hook function that applies steering."""
        steer = self.coeff * self.vector  # (hidden,)
        
        def _add(t):
            if self.positions == "all":
                return t + steer.to(t.device)
            elif self.positions == "prompt":
                if t.shape[1] == 1:
                    return t
                else:
                    t2 = t.clone()
                    t2 += steer.to(t.device)
                    return t2
            elif self.positions == "response":
                t2 = t.clone()
                t2[:, -1, :] += steer.to(t.device)
                return t2
            else:
                raise ValueError(f"Invalid positions: {self.positions}")
        
        # out may be tensor or tuple/list => normalise to tuple
        if torch.is_tensor(out):
            new_out = _add(out)
        elif isinstance(out, (tuple, list)):
            if not torch.is_tensor(out[0]):
                # unusual case – don't touch
                return out
            head = _add(out[0])
            new_out = (head, *out[1:])  # keep other entries
        else:
            return out  # unknown type – leave unchanged
        
        if self.debug:
            with torch.no_grad():
                delta = (new_out[0] if isinstance(new_out, tuple) else new_out) - (
                    out[0] if isinstance(out, (tuple, list)) else out
                )
                print(
                    "[ActivationSteerer] |delta| (mean ± std): "
                    f"{delta.abs().mean():.4g} ± {delta.std():.4g}"
                )
        return new_out
    
    # Context manager
    def __enter__(self):
        layer = self._locate_layer()
        self._handle = layer.register_forward_hook(self._hook_fn)
        return self
    
    def __exit__(self, *exc):
        self.remove()  # always clean up
    
    def remove(self):
        if self._handle:
            self._handle.remove()
            self._handle = None


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
def apply_steering(
    vector_path: str = None,  # If None, will load truthful/strategic and compute difference
    config_path: str = "configs/config_plurality_40_truthful.json",
    layer_idx: int = 20,
    coefficients: List[float] = None,
    num_trials: int = 20,
    output_dir: str = None,  # Will be set based on model name
    steering_type: str = "response",
    overwrite: bool = True,
    base_seed: int = 67,
    reasoning_level: str = "low"  # "low", "medium", "high", or None for default
):
    """
    Apply steering with varying coefficients and collect outputs.
    Uses custom ActivationSteerer class with forward hooks.
    
    Args:
        vector_path: Path to steering vector .pt file (can be all-layers or single-layer)
        config_path: Path to config file
        layer_idx: Layer to apply steering
        coefficients: List of coefficients to test (default: -5 to 5 in steps of 1)
        num_trials: Number of trials per coefficient
        output_dir: Directory to save results
        steering_type: "all", "prompt", or "response" (default: "response")
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Set output directory based on model name if not provided
    if output_dir is None:
        output_dir = f"/results/steering_vectors/results/{MODEL_DIR_NAME}/results"
    
    if coefficients is None:
        # Default: sweep truthfulness steering from 0 to 3 in steps of 0.5
        coefficients = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
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
    
    # Load per-sample vectors and compute averages, then difference on the fly
    # If vector_path is None, load truthful/strategic samples and compute averages + difference
    if vector_path is None:
        # Use Modal volume paths
        base_dir = f"/results/steering_vectors/results/{MODEL_DIR_NAME}"
        truthful_path = f"{base_dir}/truthful_response_samples.pt"
        strategic_path = f"{base_dir}/strategic_response_samples.pt"
        print(f"Loading steering vectors from Modal volume...")
        print(f"  Truthful: {truthful_path}")
        print(f"  Strategic: {strategic_path}")
        truthful_samples = torch.load(truthful_path, map_location="cuda", weights_only=False)
        strategic_samples = torch.load(strategic_path, map_location="cuda", weights_only=False)
        # Compute averages: [num_samples, layers, hidden_dim] -> [layers, hidden_dim]
        truthful_data = truthful_samples.mean(dim=0)
        strategic_data = strategic_samples.mean(dim=0)
        # Compute difference: truthful - strategic
        vector_data = truthful_data - strategic_data
        print(f"  Loaded per-sample vectors: truthful {truthful_samples.shape}, strategic {strategic_samples.shape}")
        print(f"  Computed averages: {truthful_data.shape} [layers × hidden_dim]")
    elif vector_path.startswith("/results"):
        # Modal volume path (legacy support for direct difference file)
        print(f"Loading steering vector from Modal volume: {vector_path}...")
        vector_data = torch.load(vector_path, map_location="cuda", weights_only=False)
    else:
        # Try local mount first, then Modal volume
        local_base_dir = f"/project/steering_vectors/results/{MODEL_DIR_NAME}/extraction"
        local_truthful_path = f"{local_base_dir}/truthful_response_samples.pt"
        local_strategic_path = f"{local_base_dir}/strategic_response_samples.pt"
        
        if os.path.exists(local_truthful_path) and os.path.exists(local_strategic_path):
            print(f"Loading steering vectors from local mount...")
            print(f"  Truthful: {local_truthful_path}")
            print(f"  Strategic: {local_strategic_path}")
            truthful_samples = torch.load(local_truthful_path, map_location="cuda", weights_only=False)
            strategic_samples = torch.load(local_strategic_path, map_location="cuda", weights_only=False)
            # Compute averages: [num_samples, layers, hidden_dim] -> [layers, hidden_dim]
            truthful_data = truthful_samples.mean(dim=0)
            strategic_data = strategic_samples.mean(dim=0)
            # Compute difference: truthful - strategic
            vector_data = truthful_data - strategic_data
            print(f"  Loaded per-sample vectors: truthful {truthful_samples.shape}, strategic {strategic_samples.shape}")
            print(f"  Computed averages: {truthful_data.shape} [layers × hidden_dim]")
        else:
            # Fallback: try as direct difference file
            local_vector_path = f"/project/{vector_path}" if not vector_path.startswith("/project") else vector_path
            if os.path.exists(local_vector_path):
                print(f"Loading steering vector from local mount: {local_vector_path}...")
                vector_data = torch.load(local_vector_path, map_location="cuda", weights_only=False)
            else:
                # Try Modal volume
                base_dir = f"/results/steering_vectors/results/{MODEL_DIR_NAME}"
                truthful_path = f"{base_dir}/truthful_response_samples.pt"
                strategic_path = f"{base_dir}/strategic_response_samples.pt"
                if os.path.exists(truthful_path) and os.path.exists(strategic_path):
                    print(f"Loading steering vectors from Modal volume...")
                    print(f"  Truthful: {truthful_path}")
                    print(f"  Strategic: {strategic_path}")
                    truthful_samples = torch.load(truthful_path, map_location="cuda", weights_only=False)
                    strategic_samples = torch.load(strategic_path, map_location="cuda", weights_only=False)
                    # Compute averages: [num_samples, layers, hidden_dim] -> [layers, hidden_dim]
                    truthful_data = truthful_samples.mean(dim=0)
                    strategic_data = strategic_samples.mean(dim=0)
                    # Compute difference: truthful - strategic
                    vector_data = truthful_data - strategic_data
                    print(f"  Loaded per-sample vectors: truthful {truthful_samples.shape}, strategic {strategic_samples.shape}")
                    print(f"  Computed averages: {truthful_data.shape} [layers × hidden_dim]")
                else:
                    # Last resort: try as direct difference file
                    volume_path = f"/results/steering_vectors/results/{MODEL_DIR_NAME}/{os.path.basename(vector_path)}"
                    if not os.path.exists(volume_path):
                        volume_path = f"/results/steering_vectors/results/{os.path.basename(vector_path)}"
                    print(f"Loading steering vector from Modal volume: {volume_path}...")
                    vector_data = torch.load(volume_path, map_location="cuda", weights_only=False)
    
    # Handle both all-layers format [layers × hidden_dim] and single-layer format [hidden_dim]
    if vector_data.ndim == 2:
        # All-layers format: select the specified layer
        steering_vector = vector_data[layer_idx]  # [hidden_dim]
        print(f"  Loaded/computed all-layers vector, using layer {layer_idx}")
    else:
        # Single-layer format
        steering_vector = vector_data  # [hidden_dim]
        print(f"  Loaded/computed single-layer vector")
    
    print(f"  Vector shape: {steering_vector.shape}")
    
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
            # Per cookbook: Keep system prompt simple, format instructions are in user prompt
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
    
    # Create output directories (both volume and local, organized by model name)
    os.makedirs(output_dir, exist_ok=True)
    local_output_dir = f"/project/steering_vectors/results/{MODEL_DIR_NAME}/results"
    os.makedirs(local_output_dir, exist_ok=True)
    
    # Check if summary exists and warn/error if overwrite not allowed
    summary_path_volume = os.path.join(output_dir, "summary.json")
    if not overwrite and os.path.exists(summary_path_volume):
        raise FileExistsError(
            f"Results already exist at {output_dir}. "
            f"Set overwrite=True to overwrite, or use a different output_dir."
        )
    
    all_results = {}
    
    # Print the prompt that will be used for all trials
    print(f"\n{'='*80}")
    print("PROMPT USED FOR ALL TRIALS")
    print(f"{'='*80}")
    print(neutral_prompt)
    print(f"{'='*80}\n")
    
    # Run baseline (no steering) first
    print(f"\n{'='*80}")
    print("BASELINE: No Steering Applied")
    print(f"{'='*80}")
    print("Running baseline to compare against steering results...")
    
    baseline_trials = []
    for trial in range(1, num_trials + 1):
        print(f"  Baseline trial {trial}/{num_trials}...", end=" ", flush=True)
        
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
                inputs = tokenizer(neutral_prompt, return_tensors="pt").to(model.device)
            
            # Generate WITHOUT steering (baseline)
            with torch.no_grad():
                generated = model.generate(**inputs, max_new_tokens=2048)
            
            # Decode - Extract only the generated tokens (after input)
            # inputs is a BatchEncoding (dict-like) with 'input_ids' key
            input_length = inputs['input_ids'].shape[1]
            output_tokens = generated[0][input_length:]
            num_tokens = output_tokens.shape[0]
            output_text = tokenizer.decode(
                output_tokens,
                skip_special_tokens=True
            )
            # Log token/character statistics similar to baseline_inference script
            print(f"\n    [Generated {num_tokens} tokens, {len(output_text)} chars]")
            
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
            print(f"✗ Error: {e}")
            trial_data["error"] = str(e)
        
        baseline_trials.append(trial_data)
    
    # Save baseline results
    baseline_thinkings = [t.get("thinking") for t in baseline_trials]
    baseline_outputs = [t["output"] for t in baseline_trials]
    baseline_votes = [t["vote"] for t in baseline_trials]
    
    baseline_result = {
        "coefficient": None,  # No steering applied
        "steering_type": "none",  # Baseline - no steering
        "num_trials": num_trials,
        "prompt": neutral_prompt,
        "trials": baseline_trials,
        "thinkings": baseline_thinkings,  # Internal reasoning (if separated)
        "outputs": baseline_outputs,      # Final formatted outputs (used for vote parsing)
        "votes": baseline_votes,
        "valid_votes": [v for v in baseline_votes if v is not None]
    }
    
    # Save baseline to both volume and local
    baseline_file_volume = os.path.join(output_dir, "baseline_no_steering.json")
    baseline_file_local = os.path.join(local_output_dir, "baseline_no_steering.json")
    
    with open(baseline_file_volume, 'w') as f:
        json.dump(baseline_result, f, indent=2)
    
    with open(baseline_file_local, 'w') as f:
        json.dump(baseline_result, f, indent=2)
    
    baseline_valid = len(baseline_result["valid_votes"])
    print(f"\n✓ Baseline complete: {baseline_valid}/{num_trials} valid votes")
    print(f"  Saved to volume: {baseline_file_volume}")
    print(f"  Saved to local: steering_vectors/results/{MODEL_DIR_NAME}/results/baseline_no_steering.json")
    
    # Test each coefficient
    for coef in coefficients:
        print(f"\n{'='*80}")
        print(f"Testing coefficient: {coef} (steering_type: {steering_type})")
        print(f"{'='*80}")
        
        trials = []
        
        for trial in range(1, num_trials + 1):
            print(f"  Trial {trial}/{num_trials}...", end=" ", flush=True)
            
            trial_data = {
                "trial": trial,
                "prompt": neutral_prompt,
                "output": None,
                "vote": None,
                "error": None
            }
            
            try:
                # Set seed for reproducibility (base_seed + trial number)
                seed = base_seed + trial
                torch.manual_seed(seed)
                random.seed(seed)
                np.random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                
                # Prepare inputs - Use official Transformers approach for GPT OSS
                if use_chat_template:
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
                    inputs = tokenizer(neutral_prompt, return_tensors="pt").to(model.device)
                
                # Generate with steering using context manager (forward hook approach)
                # Note: Even with coef=0, the hook is still registered (multiplies by 0)
                with ActivationSteerer(
                    model,
                    steering_vector,
                    coeff=coef,
                    layer_idx=layer_idx,
                    positions=steering_type
                ):
                    with torch.no_grad():
                        generated = model.generate(**inputs, max_new_tokens=2048)
                
                # Decode - Extract only the generated tokens (after input)
                # inputs is a BatchEncoding (dict-like) with 'input_ids' key
                input_length = inputs['input_ids'].shape[1]
                output_tokens = generated[0][input_length:]
                num_tokens = output_tokens.shape[0]
                output_text = tokenizer.decode(
                    output_tokens,
                    skip_special_tokens=True
                )
                # Log token/character statistics for steering runs
                print(f"\n    [Generated {num_tokens} tokens, {len(output_text)} chars]")
                
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
                print(f"✗ Error: {e}")
                trial_data["error"] = str(e)
            
            trials.append(trial_data)
        
        # Extract lists for backward compatibility
        thinkings = [t.get("thinking") for t in trials]
        outputs = [t["output"] for t in trials]
        votes = [t["vote"] for t in trials]
        
        # Save results for this coefficient (both volume and local)
        coef_str = f"{coef:+.1f}".replace("+", "pos").replace("-", "neg")
        coef_file_volume = os.path.join(output_dir, f"coef_{coef_str}.json")
        coef_file_local = os.path.join(local_output_dir, f"coef_{coef_str}.json")
        
        # Structure of coef_*.json file:
        # {
        #   "coefficient": float,           # Steering coefficient (e.g., -5.0, 0.0, 5.0)
        #   "num_trials": int,              # Number of trials run
        #   "steering_type": str,           # "all", "prompt", or "response"
        #   "prompt": str,                  # Full prompt text used for all trials
        #   "trials": [                     # Array of trial-by-trial results
        #     {
        #       "trial": int,               # Trial number (1-indexed)
        #       "prompt": str,               # Prompt used (same for all trials)
        #       "output": str,               # Model's generated response text
        #       "vote": str or null,        # Parsed vote choice (A, B, C, or null if parse failed)
        #       "error": str or null        # Error message if generation failed
        #     },
        #     ...
        #   ],
        #   "outputs": [str, ...],          # List of outputs (backward compatibility)
        #   "votes": [str or null, ...],     # List of votes (backward compatibility)
        #   "valid_votes": [str, ...]       # List of non-null votes
        # }
        result = {
            "coefficient": coef,
            "num_trials": num_trials,
            "steering_type": steering_type,
            "prompt": neutral_prompt,  # Full prompt used for all trials
            "trials": trials,  # Detailed trial-by-trial data
            # Backward compatibility fields
            "thinkings": thinkings,  # Internal reasoning (if separated)
            "outputs": outputs,      # Final formatted outputs (used for vote parsing)
            "votes": votes,
            "valid_votes": [v for v in votes if v is not None]
        }
        
        # Save to volume
        with open(coef_file_volume, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Save to local
        with open(coef_file_local, 'w') as f:
            json.dump(result, f, indent=2)
        
        all_results[coef] = result
        print(f"  Saved to volume: {coef_file_volume}")
        print(f"  Saved to local: steering_vectors/results/{MODEL_DIR_NAME}/results/coef_{coef_str}.json")
    
    # Save summary (both volume and local)
    summary = {
        "model": MODEL_NAME,
        "layer": layer_idx,
        "vector_path": vector_path,
        "steering_type": steering_type,
        "coefficients": coefficients,
        "num_trials": num_trials,
        "baseline": {
            "valid_votes": len(baseline_result["valid_votes"]),
            "votes": baseline_result["votes"]
        },
        "results": {str(k): {"valid_votes": len(v["valid_votes"]), "votes": v["votes"]} 
                    for k, v in all_results.items()}
    }
    
    # Save summary (both volume and local)
    summary_path_volume = os.path.join(output_dir, "summary.json")
    with open(summary_path_volume, 'w') as f:
        json.dump(summary, f, indent=2)
    
    summary_path_local = os.path.join(local_output_dir, "summary.json")
    with open(summary_path_local, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Commit volume
    results_vol.commit()
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    # Show baseline first
    baseline_valid = len(baseline_result["valid_votes"])
    print(f"Baseline (no steering): {baseline_valid}/{num_trials} valid votes")
    if baseline_valid > 0:
        from collections import Counter
        baseline_vote_counts = Counter(baseline_result["valid_votes"])
        print(f"  Vote distribution: {dict(baseline_vote_counts)}")
    
    print()  # Blank line
    
    # Show steering results
    for coef in coefficients:
        r = all_results[coef]
        valid = len(r["valid_votes"])
        print(f"Coefficient {coef:6.1f}: {valid}/{num_trials} valid votes")
        if valid > 0:
            from collections import Counter
            vote_counts = Counter(r["valid_votes"])
            print(f"  Vote distribution: {dict(vote_counts)}")
    
    print(f"\n✓ All results saved to:")
    print(f"  Modal volume: {output_dir}")
    print(f"  Local repo: {local_output_dir}")
    print(f"    (maps to: steering_vectors/results/{MODEL_DIR_NAME}/results/ in your repo root)")
    return output_dir


@app.local_entrypoint()
def main(
    vector_path: str = None,  # Will default to model-specific path
    config_path: str = "configs/config_plurality_40_truthful.json",
    layer_idx: int = 20,
    coefficients: str = "0,0.5,1,1.5,2,2.5,3",
    num_trials: int = 20,
    steering_type: str = "response",
    overwrite: bool = True,
    base_seed: int = 67,
    reasoning_level: str = "low"  # "low", "medium", "high", or "" for default
):
    """Local entrypoint to run steering experiments."""
    # Note: vector_path is now optional - if None, the function will load
    # truthful_response_avg.pt and strategic_response_avg.pt and compute the difference
    
    # Parse coefficients
    coef_list = [float(x.strip()) for x in coefficients.split(",")]
    
    # Convert empty string to None for reasoning_level
    reasoning = None if reasoning_level == "" else reasoning_level
    
    output_dir = apply_steering.remote(
        vector_path=vector_path,
        config_path=config_path,
        layer_idx=layer_idx,
        coefficients=coef_list,
        num_trials=num_trials,
        steering_type=steering_type,
        overwrite=overwrite,
        base_seed=base_seed,
        reasoning_level=reasoning
    )
    
    print(f"\n✓ Steering experiments complete!")
    print(f"Results saved to Modal volume: {output_dir}")
    
    # Download files from Modal volume to local repo
    from pathlib import Path
    
    local_output_dir = Path(f"steering_vectors/results/{MODEL_DIR_NAME}/results")
    local_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDownloading files from Modal volume to local repo...")
    # Note: reload() can only be called from within a Modal function, not from local entrypoint
    # The volume should already have the latest files after the remote function completes
    
    # Download baseline
    try:
        baseline_data = results_vol.read_file(f"steering_outputs/{MODEL_DIR_NAME}/baseline_no_steering.json")
        if isinstance(baseline_data, bytes):
            data = baseline_data
        else:
            data = b''.join(baseline_data) if hasattr(baseline_data, '__iter__') else bytes(baseline_data)
        
        with open(local_output_dir / "baseline_no_steering.json", 'wb') as f:
            f.write(data)
        print(f"  ✓ Downloaded: baseline_no_steering.json")
    except Exception as e:
        print(f"  ⚠ Could not download baseline_no_steering.json: {e}")
    
    # Download summary.json
    try:
        summary_data = results_vol.read_file(f"steering_vectors/results/{MODEL_DIR_NAME}/results/summary.json")
        # Handle both bytes and generator (streaming) responses
        if isinstance(summary_data, bytes):
            data = summary_data
        else:
            # If it's a generator/stream, read all chunks
            data = b''.join(summary_data) if hasattr(summary_data, '__iter__') else bytes(summary_data)
        
        with open(local_output_dir / "summary.json", 'wb') as f:
            f.write(data)
        print(f"  ✓ Downloaded: summary.json")
    except Exception as e:
        print(f"  ⚠ Could not download summary.json: {e}")
    
    # Download coefficient files
    for coef in coef_list:
        coef_str = f"{coef:+.1f}".replace("+", "pos").replace("-", "neg")
        filename = f"coef_{coef_str}.json"
        volume_path = f"steering_vectors/results/{MODEL_DIR_NAME}/results/{filename}"
        
        try:
            file_data = results_vol.read_file(volume_path)
            # Handle both bytes and generator (streaming) responses
            if isinstance(file_data, bytes):
                data = file_data
            else:
                # If it's a generator/stream, read all chunks
                data = b''.join(file_data) if hasattr(file_data, '__iter__') else bytes(file_data)
            
            with open(local_output_dir / filename, 'wb') as f:
                f.write(data)
            print(f"  ✓ Downloaded: {filename}")
        except Exception as e:
            print(f"  ⚠ Could not download {filename}: {e}")
    
    print(f"\n✓ Files downloaded to: {local_output_dir.absolute()}")
