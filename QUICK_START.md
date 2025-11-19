# Quick Start Guide

## Using the Template-Based Experiment Runner

### Basic Usage

```bash
# Run a simple experiment
python run_experiment.py --voting-rule plurality --instruction-type truthful --trials 10
```

### Common Examples

**Plurality Voting:**
```bash
# Truthful voting
python run_experiment.py --voting-rule plurality --instruction-type truthful --trials 10

# Strategic voting
python run_experiment.py --voting-rule plurality --instruction-type strategic --trials 10

# Neutral voting
python run_experiment.py --voting-rule plurality --instruction-type neutral --trials 10
```

**IRV Voting:**
```bash
python run_experiment.py --voting-rule irv --instruction-type truthful --trials 10
```

**Borda Count:**
```bash
python run_experiment.py --voting-rule borda --instruction-type strategic --trials 10
```

### Using Config Files

1. Copy the example config:
```bash
cp config_example.json my_config.json
```

2. Edit `my_config.json` with your settings

3. Run:
```bash
python run_experiment.py --config my_config.json
```

### Customizing Parameters

**Change model:**
```bash
python run_experiment.py --voting-rule plurality --instruction-type truthful --model gpt-4-turbo
```

**Change temperature:**
```bash
python run_experiment.py --voting-rule plurality --instruction-type truthful --temperature 0.3
```

**Custom voter information:**
```bash
python run_experiment.py --voting-rule plurality --instruction-type truthful \
  --voter-information "A=30, B=40, C=20, You control 10 votes"
```

**Custom preference profile:**
```bash
python run_experiment.py --voting-rule plurality --instruction-type truthful \
  --preference-profile "A > C > B"
```

### Output

Results are saved in the `results/` directory with:
- `{voting_rule}_{instruction_type}_{timestamp}_all_trials.txt` - All trial results
- `{voting_rule}_{instruction_type}_{timestamp}_config.json` - Configuration used

