# Quick Start Guide

## Using Independent Agents

### Basic Usage

```bash
# Run from project root
python -m src.run_independent_agents --voting-rule plurality --instruction-type truthful --num-agents 20 --trials 1
```

### Common Examples

**Plurality Voting:**
```bash
# Truthful voting
python -m src.run_independent_agents --voting-rule plurality --instruction-type truthful --num-agents 20 --trials 1

# Strategic voting
python -m src.run_independent_agents --voting-rule plurality --instruction-type strategic --num-agents 20 --trials 1
```

**IRV Voting:**
```bash
python -m src.run_independent_agents --voting-rule irv --instruction-type truthful --num-agents 20 --trials 1
```

**Borda Count:**
```bash
python -m src.run_independent_agents --voting-rule borda --instruction-type strategic --num-agents 20 --trials 1
```

### Using Config Files

1. Copy the example config:
```bash
cp configs/config_independent_agents_example.json configs/my_config.json
```

2. Edit `configs/my_config.json` with your settings

3. Run:
```bash
python -m src.run_independent_agents --config configs/my_config.json --num-agents 20
```

## Using Block Agents

```bash
# Run with CLI args
python -m src.run_block_agents --voting-rule plurality --instruction-type truthful --trials 10

# Run with config file
python -m src.run_block_agents --config configs/config_example.json
```

## Output

Results are saved in the `results/` directory with:
- `{voting_rule}_{instruction_type}_{timestamp}_all_trials.txt` - All trial results
- `{voting_rule}_{instruction_type}_{timestamp}_config.json` - Configuration used
- `{voting_rule}_{instruction_type}_{timestamp}_llm_queries.log` - All LLM queries logged
