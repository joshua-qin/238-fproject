#!/usr/bin/env python3
"""
Run voting experiments with independent agents (one agent per voter).
"""

import argparse
import json
import os
import re
from collections import Counter
from datetime import datetime
from .prompt_template import generate_prompt, VOTING_RULE_DESCRIPTIONS, INSTRUCTION_TYPES
from .llm_utils import call_gpt, ensure_directories, RESULTS_DIR, setup_logger


def load_config(config_file):
    """Load configuration from JSON file."""
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_config(config, output_file):
    """Save configuration to JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)


def parse_vote_choice(output, voting_rule):
    """Parse vote choice from agent output."""
    if voting_rule == "plurality":
        # Look for "Vote Choice: Candidate X" pattern
        match = re.search(r'\*\*Vote Choice:\*\*\s*(?:Candidate\s*)?([ABC])', output, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        # Fallback: look for just "Candidate A/B/C" in the output
        match = re.search(r'Candidate\s*([ABC])', output, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    else:
        # For IRV and Borda, look for ranking
        match = re.search(r'\*\*Ranking of Candidates:\*\*\s*([ABC])\s*>\s*([ABC])\s*>\s*([ABC])', output, re.IGNORECASE)
        if match:
            return match.group(1).upper() + " > " + match.group(2).upper() + " > " + match.group(3).upper()
        # Fallback: look for "A > B > C" pattern
        match = re.search(r'([ABC])\s*>\s*([ABC])\s*>\s*([ABC])', output, re.IGNORECASE)
        if match:
            return match.group(1).upper() + " > " + match.group(2).upper() + " > " + match.group(3).upper()
    return None


def aggregate_votes(votes, voting_rule):
    """Aggregate votes from multiple agents."""
    if voting_rule == "plurality":
        # Count votes for each candidate
        vote_counts = Counter(votes)
        return {
            "A": vote_counts.get("A", 0),
            "B": vote_counts.get("B", 0),
            "C": vote_counts.get("C", 0),
            "total": len(votes),
            "winner": vote_counts.most_common(1)[0][0] if vote_counts else None
        }
    else:
        # For IRV and Borda, return the distribution of rankings
        ranking_counts = Counter(votes)
        return {
            "rankings": dict(ranking_counts),
            "total": len(votes),
            "most_common": ranking_counts.most_common(5)
        }


def modify_voter_information_for_individual(voter_information, num_agents, other_votes):
    """
    Modify voter information to reflect individual voter context.
    
    Args:
        voter_information: Original voter information string
        num_agents: Number of agents in the block
        other_votes: Dict with votes from other voters (not in this block)
    
    Returns:
        Modified voter information string
    """
    # Replace "You control a block of X votes" with individual context
    modified = voter_information
    
    # Try to find and replace block control language
    modified = re.sub(
        r'You control (?:a block of |the remaining block of )?(\d+) votes?',
        f'There are {num_agents} voters with the same preferences as you (including yourself). '
        f'You are one individual voter in this group. '
        f'You do not know how the other {num_agents - 1} voters in your group will vote.',
        modified,
        flags=re.IGNORECASE
    )
    
    # Add information about other voters' votes
    if other_votes:
        other_info = "You have complete information about how other voters (outside your group) will vote:\n"
        for candidate, count in other_votes.items():
            other_info += f"- Candidate {candidate} will receive {count} votes\n"
        modified = other_info + "\n" + modified
    
    return modified


def run_independent_agents(
    prompt_template,
    model,
    temperature,
    num_agents,
    voting_rule,
    voter_information,
    other_votes=None,
    num_trials=1,
    logger=None
):
    """
    Run multiple independent agents and aggregate their votes.
    
    Args:
        prompt_template: Base prompt template
        model: GPT model to use
        temperature: Temperature for generation
        num_agents: Number of independent agents to run
        voting_rule: Voting rule being used
        voter_information: Original voter information
        other_votes: Dict with votes from other voters
        num_trials: Number of trials to run
    
    Returns:
        List of trial results, each containing agent outputs and aggregated votes
    """
    all_trial_results = []
    
    for trial in range(1, num_trials + 1):
        print(f"\n{'='*80}")
        print(f"TRIAL {trial}/{num_trials}")
        print(f"{'='*80}")
        
        # Modify prompt for individual voter context
        modified_voter_info = modify_voter_information_for_individual(
            voter_information,
            num_agents,
            other_votes
        )
        
        # Replace voter information in prompt - use regex for more robust matching
        # Find the VOTER INFORMATION section and replace it
        pattern = r'VOTER INFORMATION\n.*?(?=\n\nYOUR PROFILE|\nYOUR PROFILE)'
        replacement = f"VOTER INFORMATION\n{modified_voter_info}"
        prompt = re.sub(pattern, replacement, prompt_template, flags=re.DOTALL)
        
        # Also remove any remaining references to "block of X votes" in the entire prompt
        prompt = re.sub(
            r'You control (?:a block of |the remaining block of )?(\d+) votes?',
            f'You are one individual voter among {num_agents} voters with the same preferences',
            prompt,
            flags=re.IGNORECASE
        )
        prompt = re.sub(
            r'casting (?:all |your )?(\d+) (?:of )?(?:your )?votes?',
            'casting your vote',
            prompt,
            flags=re.IGNORECASE
        )
        prompt = re.sub(
            r'(\d+) votes? (?:for|to)',
            'your vote for',
            prompt,
            flags=re.IGNORECASE
        )
        
        # Run all agents independently
        agent_outputs = []
        agent_votes = []
        
        print(f"Running {num_agents} independent agents...")
        for agent_id in range(1, num_agents + 1):
            print(f"  Agent {agent_id}/{num_agents}...", end=" ", flush=True)
            try:
                query_id = f"Trial{trial}_Agent{agent_id}"
                output = call_gpt(prompt, model, temperature, logger, query_id)
                agent_outputs.append(output)
                
                # Parse vote choice
                vote = parse_vote_choice(output, voting_rule)
                agent_votes.append(vote)
                
                if vote:
                    print(f"✓ ({vote})")
                else:
                    print(f"⚠ (could not parse)")
            except Exception as e:
                print(f"✗ Error: {e}")
                agent_outputs.append(f"[ERROR: {e}]")
                agent_votes.append(None)
        
        # Aggregate votes
        valid_votes = [v for v in agent_votes if v is not None]
        aggregated = aggregate_votes(valid_votes, voting_rule)
        
        trial_result = {
            "trial": trial,
            "agent_outputs": agent_outputs,
            "agent_votes": agent_votes,
            "aggregated": aggregated,
            "num_agents": num_agents,
            "valid_votes": len(valid_votes)
        }
        
        all_trial_results.append(trial_result)
        
        # Print summary
        print(f"\nTrial {trial} Summary:")
        print(f"  Valid votes: {len(valid_votes)}/{num_agents}")
        if voting_rule == "plurality":
            print(f"  Vote distribution: A={aggregated['A']}, B={aggregated['B']}, C={aggregated['C']}")
            print(f"  Winner: Candidate {aggregated['winner']}")
        else:
            print(f"  Most common rankings:")
            for ranking, count in aggregated['most_common'][:3]:
                print(f"    {ranking}: {count} votes")
    
    return all_trial_results


def save_results(results, output_base, voting_rule):
    """Save results to file."""
    ensure_directories()
    
    output_file = os.path.join(RESULTS_DIR, f"{output_base}_independent_agents.txt")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Independent Agents Experiment Results\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Voting Rule: {voting_rule}\n")
        f.write(f"Number of Agents: {results[0]['num_agents']}\n")
        f.write(f"Number of Trials: {len(results)}\n\n")
        
        for trial_result in results:
            f.write(f"{'='*80}\n")
            f.write(f"TRIAL {trial_result['trial']}\n")
            f.write(f"{'-'*80}\n\n")
            
            # Aggregated results
            f.write("AGGREGATED RESULTS:\n")
            if voting_rule == "plurality":
                f.write(f"  Candidate A: {trial_result['aggregated']['A']} votes\n")
                f.write(f"  Candidate B: {trial_result['aggregated']['B']} votes\n")
                f.write(f"  Candidate C: {trial_result['aggregated']['C']} votes\n")
                f.write(f"  Winner: Candidate {trial_result['aggregated']['winner']}\n")
            else:
                f.write(f"  Most common rankings:\n")
                for ranking, count in trial_result['aggregated']['most_common'][:5]:
                    f.write(f"    {ranking}: {count} votes\n")
            
            f.write(f"\nValid votes: {trial_result['valid_votes']}/{trial_result['num_agents']}\n\n")
            
            # Individual agent outputs
            f.write("INDIVIDUAL AGENT OUTPUTS:\n")
            f.write(f"{'-'*80}\n")
            for i, (output, vote) in enumerate(zip(trial_result['agent_outputs'], trial_result['agent_votes']), 1):
                f.write(f"\nAgent {i}:\n")
                f.write(f"Vote: {vote if vote else 'PARSE ERROR'}\n")
                f.write(f"Output:\n{output}\n")
                f.write(f"\n{'-'*80}\n")
            
            f.write(f"\n\n")
    
    print(f"\nResults saved to: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Run voting experiments with independent agents (one agent per voter)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with command-line arguments
  python run_independent_agents.py --voting-rule plurality --instruction-type truthful --num-agents 20
  
  # Run with config file
  python run_independent_agents.py --config config.json --num-agents 20
  
  # Run multiple trials
  python run_independent_agents.py --config config.json --num-agents 20 --trials 5
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
    
    # Agent parameters
    parser.add_argument(
        '--num-agents',
        type=int,
        required=True,
        help='Number of independent agents (one per voter)'
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
    
    # Other votes (for context)
    parser.add_argument(
        '--other-votes',
        help='JSON string with votes from other voters, e.g., \'{"A": 35, "B": 45}\''
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
    
    # Override with command-line arguments
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
        if os.path.isfile(args.candidates):
            with open(args.candidates, 'r') as f:
                config['candidates'] = json.load(f)
        else:
            config['candidates'] = json.loads(args.candidates)
    if args.voter_information:
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
    if args.other_votes:
        config['other_votes'] = json.loads(args.other_votes)
    
    # Validate required parameters
    if 'voting_rule' not in config:
        parser.error("--voting-rule is required (or specify in config file)")
    if 'instruction_type' not in config:
        parser.error("--instruction-type is required (or specify in config file)")
    
    num_agents = args.num_agents
    
    # Generate base prompt
    print("Generating prompt template...")
    prompt_template = generate_prompt(
        voting_rule=config.get('voting_rule', 'plurality'),
        candidates=config.get('candidates'),
        voter_information=config.get('voter_information'),
        voter_profile=config.get('voter_profile'),
        preference_profile=config.get('preference_profile'),
        instruction_type=config.get('instruction_type', 'truthful'),
        total_voters=config.get('total_voters', 100)
    )
    
    # Get original voter information for modification
    original_voter_info = config.get('voter_information', "")
    
    # Get other votes
    other_votes = config.get('other_votes')
    
    # Generate output filename
    if args.output:
        output_base = os.path.splitext(args.output)[0]
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = f"{config['voting_rule']}_{config['instruction_type']}_{num_agents}agents_{timestamp}"
    
    # Save config used for this run
    ensure_directories()
    config_file = os.path.join(RESULTS_DIR, f"{output_base}_config.json")
    config['num_agents'] = num_agents
    save_config(config, config_file)
    print(f"Configuration saved to: {config_file}")
    
    # Set up logging for LLM queries
    log_file = os.path.join(RESULTS_DIR, f"{output_base}_llm_queries.log")
    logger = setup_logger(log_file)
    print(f"LLM query logging enabled: {log_file}")
    
    # Run independent agents
    print(f"\nUsing model: {config.get('model', 'gpt-4o')}")
    print(f"Temperature: {config.get('temperature', 0.7)}")
    print(f"Trials: {config.get('trials', 1)}")
    print(f"Voting rule: {config['voting_rule']}")
    print(f"Instruction type: {config['instruction_type']}")
    print(f"Number of independent agents: {num_agents}")
    
    results = run_independent_agents(
        prompt_template,
        config.get('model', 'gpt-4o'),
        config.get('temperature', 0.7),
        num_agents,
        config['voting_rule'],
        original_voter_info,
        other_votes,
        config.get('trials', 1),
        logger
    )
    
    # Save results
    save_results(results, output_base, config['voting_rule'])
    
    # Print final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    if config['voting_rule'] == "plurality":
        # Aggregate across all trials
        all_votes = []
        for trial_result in results:
            all_votes.extend([v for v in trial_result['agent_votes'] if v in ['A', 'B', 'C']])
        
        vote_counts = Counter(all_votes)
        print(f"Total votes across all trials: {len(all_votes)}")
        print(f"Overall distribution:")
        print(f"  Candidate A: {vote_counts.get('A', 0)} votes ({vote_counts.get('A', 0)/len(all_votes)*100:.1f}%)")
        print(f"  Candidate B: {vote_counts.get('B', 0)} votes ({vote_counts.get('B', 0)/len(all_votes)*100:.1f}%)")
        print(f"  Candidate C: {vote_counts.get('C', 0)} votes ({vote_counts.get('C', 0)/len(all_votes)*100:.1f}%)")


if __name__ == "__main__":
    main()

