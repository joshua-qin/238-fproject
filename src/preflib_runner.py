import os
import json
import argparse
from itertools import permutations

import os
os.makedirs("results", exist_ok=True)

# -----------------------------------------
# Imports from your project structure
# -----------------------------------------
from winner_rules import irv_winner, borda_winner, plurality_winner
from llm_utils import run_trials
from prompt_template import generate_prompt_preflib


# ---------------------------------------------------
# 1. Parse the .soc file into (count, ranking) blocks
# ---------------------------------------------------
def parse_soc(path):
    blocks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            count_str, ranking_str = line.split(":")
            count = int(count_str.strip())
            ranking = list(map(int, ranking_str.strip().split(",")))
            blocks.append((count, ranking))
    return blocks


# ---------------------------------------------------
# 2. Check for profitable deviations under rule
# ---------------------------------------------------
def has_profitable_deviation(target_block, other_blocks, rule, num_candidates):
    truthful_ranking = target_block[1]

    # truthful outcome
    if rule == "irv":
        truthful_winner = irv_winner([target_block] + other_blocks)
    elif rule == "borda":
        truthful_winner = borda_winner([target_block] + other_blocks)
    elif rule == "plurality":
        truthful_winner = plurality_winner([target_block] + other_blocks)
    else:
        raise ValueError(f"Unknown rule {rule}")

    best = truthful_winner
    better_exists = False

    for perm in permutations(range(1, num_candidates + 1)):
        perm_list = list(perm)
        if perm_list == truthful_ranking:
            continue

        new_blocks = [(target_block[0], perm_list)] + other_blocks

        if rule == "irv":
            w = irv_winner(new_blocks)
        elif rule == "borda":
            w = borda_winner(new_blocks)
        else:
            w = plurality_winner(new_blocks)

        # Did the voter prefer w over best?
        if truthful_ranking.index(w) < truthful_ranking.index(best):
            better_exists = True
            best = w

    return better_exists, best


# ---------------------------------------------------
# 3. Master experiment runner
# ---------------------------------------------------
def run_preflib_experiment(args):

    blocks = parse_soc(args.soc_file)
    num_candidates = len(blocks[0][1])
    total_voters = sum(b[0] for b in blocks)

    print(f"Parsed {len(blocks)} blocks from {args.soc_file}")
    print(f"Total voters: {total_voters}, Candidates: {num_candidates}")

    results = []

    for i, block in enumerate(blocks):
        other_blocks = blocks[:i] + blocks[i+1:]

        exists, dev_winner = has_profitable_deviation(
            block,
            other_blocks,
            args.voting_rule,
            num_candidates
        )

        print(f"  Profitable deviation exists: {exists}")


        if not exists:
            continue

        # Build the prompt
        prompt = generate_prompt_preflib(
            voting_rule=args.voting_rule,
            voter_profile=f"Block {i+1}: count {block[0]}, ranking {block[1]}",
            preference_profile=blocks,
            instruction_type=args.instruction_type,
        )

        # Automatic name for model output file (placed in results/)
        output_base = (
            f"{os.path.basename(args.soc_file)}_"
            f"block{i+1}_"
            f"{args.voting_rule}_"
            f"{args.model}_"
            f"{args.instruction_type}"
        )

        print(f"\n=== Running block {i+1} → {output_base} ===")

        outputs = run_trials(
            prompt_text=prompt,
            model=args.model,
            temperature=args.temperature,
            num_trials=args.trials,
            output_base=output_base,
            logger=None,
            use_reasoning=args.use_reasoning,
            reasoning_effort=args.reasoning_effort
        )

        results.append({
            "block": i + 1,
            "truthful_ranking": block[1],
            "best_deviation_winner": dev_winner,
            "outputs": outputs
        })

    return results


# ---------------------------------------------------
# 4. Command line interface
# ---------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--soc-file", required=True,
                        help="Path to .soc file")

    parser.add_argument("--voting-rule", required=True,
                        choices=["plurality", "borda", "irv"])

    parser.add_argument("--model", required=True,
                        help="OpenAI model (e.g. gpt-5.1, gpt-5.1-mini)")

    parser.add_argument("--instruction-type", default="truthful")

    # parser.add_argument("--candidates", nargs="+",
    #                     help="Candidate labels (e.g. 1 2 3 4)", required=True)

    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--trials", type=int, default=1)

    parser.add_argument("--use-reasoning", action="store_true")
    parser.add_argument("--reasoning-effort", default="high")

    args = parser.parse_args()
    run_preflib_experiment(args)
