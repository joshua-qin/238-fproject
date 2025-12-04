# Voting Experiment Project

A research project exploring strategic voting behavior using Large Language Model (LLM) agents. This project investigates how LLMs behave as voting agents under different voting rules, instruction types, and strategic scenarios.

## Overview

This project simulates voting scenarios where LLM agents act as voters, making decisions under various voting rules (Plurality, Instant Runoff Voting, Borda Count) and instruction types (truthful, neutral, strategic). The project includes tools for:

- Running voting experiments with LLM agents
- Finding strategic misreporting opportunities in voting profiles
- Analyzing manipulability of voting rules
- Visualizing experimental results
- Conducting activation steering experiments

## Features

### Voting Rules
- **Plurality**: Candidate with most first-place votes wins
- **Instant Runoff Voting (IRV)**: Sequential elimination until majority winner
- **Borda Count**: Points-based ranking system

### Instruction Types
- **Truthful**: Agents report their true preferences
- **Neutral**: Agents receive neutral instructions without strategic guidance
- **Strategic**: Agents are instructed to strategically misreport if beneficial

### LLM Models
- GPT-4o
- GPT-5.1 (with optional reasoning API support: high, medium)

### Experiment Types
- **Block Agents**: One agent controls a block of voters with identical preferences
- **Independent Agents**: Each agent represents a single voter
- **Preflib Integration**: Run experiments on real-world voting profiles from Preflib

## Project Structure

```
.
├── src/                          # Source code
│   ├── run_block_agents.py       # Block voting experiments
│   ├── run_block_agent_preflib.py # Preflib block experiments
│   ├── run_independent_agents.py # Independent agent experiments
│   ├── preflib_runner.py         # Preflib profile runner
│   ├── find_manipulable.py       # Find manipulable profiles
│   ├── llm_utils.py              # LLM utilities with logging
│   ├── prompt_template.py        # Prompt generation templates
│   ├── winner_rules.py           # Voting rule implementations
│   └── preflib.py                # Preflib file parsing
├── configs/                      # Configuration files
│   ├── config_borda_example.json
│   ├── config_borda_minimal.json
│   ├── config_irv_example.json
│   ├── config_plurality_example.json
│   └── config_plurality_40_truthful.json
├── prompts/                      # Legacy prompt templates
├── prompts_ext/                  # Extended prompt templates
├── preflib/                      # Preflib voting profile data
├── results/                      # Experiment results
├── test_files/                   # Test voting profiles
├── steering_vectors/             # Activation steering experiments
│   ├── documentation/
│   ├── scripts/
│   └── results/
├── analyze_irv_strategy.py      # Find IRV strategic opportunities
├── analyze_borda_strategy.py     # Find Borda strategic opportunities
├── analyze_plurality_strategy.py # Find Plurality strategic opportunities
├── visualize_results.py          # Generate result visualizations
└── requirements.txt              # Python dependencies
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up OpenAI API Key

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or add it to your shell configuration file (`.bashrc`, `.zshrc`, etc.).

## Usage

### Block Agents Experiments

Run experiments where one agent controls a block of voters:

```bash
# Using CLI arguments
python -m src.run_block_agents \
    --voting-rule borda \
    --instruction-type strategic \
    --model gpt-5.1 \
    --trials 10

# Using config file
python -m src.run_block_agents --config configs/config_borda_example.json

# With reasoning API
python -m src.run_block_agents \
    --config configs/config_borda_example.json \
    --reasoning medium
```

### Independent Agents Experiments

Run experiments where each agent represents a single voter:

```bash
# Using CLI arguments
python -m src.run_independent_agents \
    --voting-rule plurality \
    --instruction-type strategic \
    --num-agents 20 \
    --trials 1

# Using config file
python -m src.run_independent_agents \
    --config configs/config_example.json \
    --num-agents 20
```

### Preflib Experiments

Run experiments on Preflib voting profiles:

```bash
python -m src.run_block_agent_preflib \
    --profile preflib/netflix.soc \
    --voting-rule irv \
    --instruction-type strategic \
    --block-index 13 \
    --model gpt-5.1 \
    --trials 10
```

### Finding Strategic Opportunities

Generate voting profiles with strategic misreporting opportunities:

```bash
# IRV strategic profile
python analyze_irv_strategy.py

# Borda Count strategic profile
python analyze_borda_strategy.py

# Plurality strategic profile
python analyze_plurality_strategy.py
```

These scripts generate `.soc` files (e.g., `modified_profile.soc`, `borda_profile.soc`) with strategic opportunities.

### Finding Manipulable Profiles

Scan Preflib profiles for manipulable blocks:

```bash
python -m src.find_manipulable preflib/ manipulable_results.txt
```

This identifies which profiles have blocks that can benefit from strategic misreporting under each voting rule.

### Visualizing Results

Generate bar charts comparing model behavior across voting rules:

```bash
python visualize_results.py
```

This creates comparison charts (`plurality_comparison.png`, `irv_comparison.png`, `borda_comparison.png`) showing the frequency of beneficial misreports across different models and instruction types.

## Configuration Files

Configuration files are JSON files that specify experiment parameters:

```json
{
  "voting_rule": "borda",
  "instruction_type": "strategic",
  "model": "gpt-5.1",
  "temperature": 0.7,
  "num_trials": 10,
  "candidates": [...],
  "voter_profile": {...},
  "preference_profile": {...}
}
```

Key fields:
- `voting_rule`: "plurality", "irv", or "borda"
- `instruction_type`: "truthful", "neutral", or "strategic"
- `model`: "gpt-4o" or "gpt-5.1"
- `reasoning`: Optional, "high" or "medium" for reasoning API
- `num_trials`: Number of experimental trials

## Results

All results are saved in the `results/` directory with the following naming convention:

- **Experiment outputs**: `{profile}_{block}_{rule}_{model}_{instruction}_{reasoning}_all_trials.txt`
- **Configuration**: `{experiment_name}_config.json`
- **LLM queries log**: `{experiment_name}_llm_queries.log`

Example: `netflix.soc_block13_irv_gpt-5.1_strategichigh_all_trials.txt`

## Vector Steering Experiments

Run activation steering experiments to study how strategic manipulation affects voting behavior:

```bash
# Step 1: Extract steering vector
modal run steering_vectors/scripts/modal_extract_steering_vector.py

# Step 2: Apply steering with varying coefficients
modal run steering_vectors/scripts/modal_apply_steering.py
```

See `steering_vectors/documentation/STEERING_USAGE.md` for detailed instructions and `steering_vectors/documentation/STEERING_SETUP.md` for setup information.

## Voting Profile Format (.soc)

The project uses the `.soc` (soc format) for voting profiles:

```
# NUMBER ALTERNATIVES: 15
# NUMBER VOTERS: 100
# ALTERNATIVE NAME 1: Candidate 1
# ALTERNATIVE NAME 2: Candidate 2
...
20: 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
15: 2,1,3,4,5,6,7,8,9,10,11,12,13,14,15
...
```

Each line represents a block: `{count}: {ranking}` where `count` is the number of voters with that ranking.

## Documentation

- `docs/QUICK_START.md` - Quick start guide for basic experiments
- `steering_vectors/documentation/STEERING_USAGE.md` - Guide for vector steering experiments
- `steering_vectors/documentation/STEERING_SETUP.md` - Setup and overview of vector steering

## Examples

### Example 1: Run Borda Count Strategic Experiment

```bash
python -m src.run_block_agents \
    --config configs/config_borda_example.json \
    --model gpt-5.1 \
    --trials 10
```

### Example 2: Find IRV Strategic Profile

```bash
python analyze_irv_strategy.py
# Output: modified_profile.soc with strategic opportunity
```

### Example 3: Analyze Preflib Profile Manipulability

```bash
python -m src.find_manipulable preflib/ manipulable_results.txt
# Output: List of manipulable profiles for each voting rule
```

## Contributing

When adding new features:
1. Update relevant configuration files
2. Add prompt templates if needed
3. Update this README
4. Ensure results are properly logged

## License

[Add license information here]
