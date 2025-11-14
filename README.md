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

