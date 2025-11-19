# Voting Experiment Project

A research project exploring voting behavior using LLM agents.

## Project Structure

```
.
├── src/                    # Source code
│   ├── run_block_agents.py      # Block voting experiments
│   ├── run_independent_agents.py # Independent agent experiments
│   ├── llm_utils.py             # LLM utilities with logging
│   └── prompt_template.py       # Prompt generation templates
├── configs/                # Configuration files
│   ├── config_example.json
│   ├── config_borda_neutral.json
│   └── config_independent_agents_example.json
├── prompts/               # Prompt templates (legacy)
├── results/               # Experiment results
├── docs/                  # Documentation
│   ├── README.md
│   └── QUICK_START.md
├── requirements.txt       # Python dependencies
└── ideas.txt             # Project notes
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Independent Agents (One Agent Per Voter)

```bash
# Run from project root
python -m src.run_independent_agents --voting-rule plurality --instruction-type strategic --num-agents 20 --trials 1

# Or with config file
python -m src.run_independent_agents --config configs/config_example.json --num-agents 20
```

### Block Agents (One Agent Controls Block of Votes)

```bash
# Run from project root
python -m src.run_block_agents --voting-rule plurality --instruction-type truthful --trials 10

# Or with config file
python -m src.run_block_agents --config configs/config_example.json
```

## Features

- **Template-based prompts**: Generate prompts from configurable templates
- **Multiple voting rules**: Plurality, IRV, Borda
- **Multiple instruction types**: Truthful, neutral, strategic
- **LLM query logging**: All queries logged to `results/*_llm_queries.log`
- **Configurable experiments**: Use JSON config files or CLI arguments

## Documentation

See `docs/QUICK_START.md` for quick start guide.

## Results

All results are saved in the `results/` directory with:
- Experiment outputs: `{experiment_name}_all_trials.txt` or `{experiment_name}_independent_agents.txt`
- Configuration: `{experiment_name}_config.json`
- LLM queries log: `{experiment_name}_llm_queries.log`
