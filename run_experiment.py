#!/usr/bin/env python3
"""
Run voting experiments with configurable hyperparameters.
"""

import argparse
import json
import os
from datetime import datetime
from prompt_template import generate_prompt, VOTING_RULE_DESCRIPTIONS, INSTRUCTION_TYPES
from run_prompt import call_gpt, run_trials, ensure_directories, RESULTS_DIR


def load_config(config_file):
    """Load configuration from JSON file."""
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_config(config, output_file):
    """Save configuration to JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Run voting experiments with configurable hyperparameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with command-line arguments
  python run_experiment.py --voting-rule plurality --instruction-type truthful --trials 10
  
  # Run with config file
  python run_experiment.py --config config.json
  
  # Run with config file and override some parameters
  python run_experiment.py --config config.json --model gpt-4-turbo --trials 5
        """
    )
    
    # Configuration file option
    parser.add_argument(
        '--config',
        '-c',
        help='Path to JSON configuration file'
    )
    
    # Model parameters
    parser.add_argument(
        '--model',
        '-m',
        default='gpt-4o',
        help='GPT model to use (default: gpt-4o)'
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
    
    # Voting rule
    parser.add_argument(
        '--voting-rule',
        choices=list(VOTING_RULE_DESCRIPTIONS.keys()),
        help='Voting rule: plurality, irv, or borda'
    )
    
    # Instruction type
    parser.add_argument(
        '--instruction-type',
        choices=list(INSTRUCTION_TYPES.keys()),
        help='Instruction type: truthful, neutral, or strategic'
    )
    
    # Candidates (as JSON string or file)
    parser.add_argument(
        '--candidates',
        help='JSON string or path to JSON file with candidate descriptions (keys: A, B, C)'
    )
    
    # Voter information
    parser.add_argument(
        '--voter-information',
        help='Voter information string (or path to file)'
    )
    
    # Voter profile
    parser.add_argument(
        '--voter-profile',
        help='Voter political ideology description'
    )
    
    # Preference profile
    parser.add_argument(
        '--preference-profile',
        help='Preference profile string (e.g., "C > A > B")'
    )
    
    # Total voters
    parser.add_argument(
        '--total-voters',
        type=int,
        default=100,
        help='Total number of voters (default: 100)'
    )
    
    # Output
    parser.add_argument(
        '--output',
        '-o',
        help='Output file path (default: auto-generated)'
    )
    
    args = parser.parse_args()
    
    # Load config from file if provided
    config = {}
    if args.config:
        config = load_config(args.config)
    
    # Override with command-line arguments (command-line takes precedence)
    if args.voting_rule:
        config['voting_rule'] = args.voting_rule
    if args.instruction_type:
        config['instruction_type'] = args.instruction_type
    if args.model:
        config['model'] = args.model
    if args.temperature is not None:
        config['temperature'] = args.temperature
    if args.trials:
        config['trials'] = args.trials
    if args.candidates:
        # Try to load as JSON file first, then as JSON string
        if os.path.isfile(args.candidates):
            with open(args.candidates, 'r') as f:
                config['candidates'] = json.load(f)
        else:
            config['candidates'] = json.loads(args.candidates)
    if args.voter_information:
        # Check if it's a file path
        if os.path.isfile(args.voter_information):
            with open(args.voter_information, 'r') as f:
                config['voter_information'] = f.read()
        else:
            config['voter_information'] = args.voter_information
    if args.voter_profile:
        config['voter_profile'] = args.voter_profile
    if args.preference_profile:
        config['preference_profile'] = args.preference_profile
    if args.total_voters:
        config['total_voters'] = args.total_voters
    
    # Validate required parameters
    if 'voting_rule' not in config:
        parser.error("--voting-rule is required (or specify in config file)")
    if 'instruction_type' not in config:
        parser.error("--instruction-type is required (or specify in config file)")
    
    # Generate prompt
    print("Generating prompt from configuration...")
    prompt = generate_prompt(
        voting_rule=config.get('voting_rule', 'plurality'),
        candidates=config.get('candidates'),
        voter_information=config.get('voter_information'),
        voter_profile=config.get('voter_profile'),
        preference_profile=config.get('preference_profile'),
        instruction_type=config.get('instruction_type', 'truthful'),
        total_voters=config.get('total_voters', 100)
    )
    
    # Generate output filename
    if args.output:
        output_base = os.path.splitext(args.output)[0]
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = f"{config['voting_rule']}_{config['instruction_type']}_{timestamp}"
    
    # Save config used for this run
    ensure_directories()
    config_file = os.path.join(RESULTS_DIR, f"{output_base}_config.json")
    save_config(config, config_file)
    print(f"Configuration saved to: {config_file}")
    
    # Run trials
    print(f"\nUsing model: {config.get('model', 'gpt-4o')}")
    print(f"Temperature: {config.get('temperature', 0.7)}")
    print(f"Trials: {config.get('trials', 1)}")
    print(f"Voting rule: {config['voting_rule']}")
    print(f"Instruction type: {config['instruction_type']}")
    
    outputs = run_trials(
        prompt,
        config.get('model', 'gpt-4o'),
        config.get('temperature', 0.7),
        config.get('trials', 1),
        output_base
    )
    
    # Print summary
    print("\n" + "="*80)
    if config.get('trials', 1) == 1:
        print("OUTPUT:")
        print("="*80)
        print(outputs[0])
    else:
        print(f"SUMMARY: {config.get('trials', 1)} trials completed")
        print("="*80)
        for i, output in enumerate(outputs, 1):
            print(f"\nTrial {i}:")
            print("-"*80)
            print(output[:500] + "..." if len(output) > 500 else output)


if __name__ == "__main__":
    main()

