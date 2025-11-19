# Voting Experiment Runner

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or create a `.env` file with:
```
OPENAI_API_KEY=your-api-key-here
```

## Usage

Run a prompt file through a GPT model:

```bash
python run_prompt.py plurality_prompt.txt --model gpt-4o --output result.txt
```

Or use a prompt directly:

```bash
python run_prompt.py "Your prompt text here" --model gpt-4o
```

### Arguments

- `prompt`: Path to prompt file or prompt text directly
- `--model` / `-m`: GPT model to use (default: gpt-4o)
- `--output` / `-o`: Output file path (default: auto-generated with timestamp)
- `--temperature` / `-t`: Temperature for generation (default: 0.7)
- `--trials` / `-n`: Number of trials to run (default: 1)

### Examples

```bash
# Basic usage with prompt file
python run_prompt.py plurality_prompt.txt

# Specify model and output file
python run_prompt.py plurality_prompt.txt --model gpt-4o --output my_result.txt

# Use different temperature
python run_prompt.py plurality_prompt.txt --temperature 0.3

# Use prompt text directly
python run_prompt.py "What is 2+2?" --model gpt-4o

# Run multiple trials (creates a directory with individual trial files and a combined file)
python run_prompt.py plurality_prompt.txt --trials 5

# Run 10 trials with custom output name
python run_prompt.py plurality_prompt.txt --trials 10 --output experiment1
```

## New: Template-Based Experiment Runner

The `run_experiment.py` script provides a more flexible, template-based approach with configurable hyperparameters.

### Quick Start

```bash
# Run with command-line arguments
python run_experiment.py --voting-rule plurality --instruction-type truthful --trials 10

# Run with config file
python run_experiment.py --config config_example.json

# Override config file parameters
python run_experiment.py --config config_example.json --model gpt-4-turbo --trials 5
```

### Configuration Options

All hyperparameters can be set via command-line arguments or a JSON config file:

- **Model**: `--model` / `-m` (default: gpt-4o)
- **Temperature**: `--temperature` / `-t` (default: 0.7)
- **Trials**: `--trials` / `-n` (default: 1)
- **Voting Rule**: `--voting-rule` (choices: plurality, irv, borda)
- **Instruction Type**: `--instruction-type` (choices: truthful, neutral, strategic)
- **Candidates**: `--candidates` (JSON string or file path)
- **Voter Information**: `--voter-information` (string or file path)
- **Voter Profile**: `--voter-profile` (political ideology description)
- **Preference Profile**: `--preference-profile` (e.g., "C > A > B")
- **Total Voters**: `--total-voters` (default: 100)
- **Output**: `--output` / `-o` (auto-generated if not specified)

### Examples

```bash
# Plurality voting, truthful instructions, 10 trials
python run_experiment.py --voting-rule plurality --instruction-type truthful --trials 10

# IRV voting, strategic instructions, custom model
python run_experiment.py --voting-rule irv --instruction-type strategic --model gpt-4-turbo --trials 5

# Borda count with custom voter information
python run_experiment.py --voting-rule borda --instruction-type neutral \
  --voter-information "You have complete information: A=30, B=40, C=20, You control 10 votes" \
  --preference-profile "C > A > B" --trials 10

# Using config file
python run_experiment.py --config config_example.json
```

### Config File Format

See `config_example.json` for a complete example. The config file can include:

```json
{
  "model": "gpt-4o",
  "temperature": 0.7,
  "trials": 10,
  "voting_rule": "plurality",
  "instruction_type": "truthful",
  "total_voters": 100,
  "candidates": {
    "A": "Candidate A description...",
    "B": "Candidate B description...",
    "C": "Candidate C description..."
  },
  "voter_information": "Voter information string...",
  "voter_profile": "Political ideology description...",
  "preference_profile": "C > A > B"
}
```

Command-line arguments override config file values.

