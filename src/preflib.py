import os
import re
import csv
import itertools
from openai import OpenAI

client = OpenAI()


############################################################
#  1. PRELIB PARSER
############################################################

def parse_preflib_file(path):
    """
    Reads a Preflib .soc/.toc file.
    Returns list of (count, [ranking]).
    """
    blocks = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            match = re.match(r"(\d+):\s*(.*)", line)
            if not match:
                continue

            count = int(match.group(1))
            ranking = list(map(int, match.group(2).split(",")))
            blocks.append((count, ranking))

    return blocks


############################################################
#  2. IRV (Instant Runoff Voting)
############################################################

def irv_winner(profile, num_candidates):
    """
    profile = list of (count, ranking_list)
    """
    eliminated = set()

    while True:
        scores = {c: 0 for c in range(1, num_candidates + 1)}

        for count, ranking in profile:
            for c in ranking:
                if c not in eliminated:
                    scores[c] += count
                    break

        active = [c for c in scores if c not in eliminated]
        if len(active) == 1:
            return active[0]

        loser = min(active, key=lambda c: scores[c])
        eliminated.add(loser)

def plurality_winner(ballots):
    """
    ballots: list of lists
        each ballot is a ranking like ["A","C","B"] (first choice first)

    returns: candidate with most first-choice votes
    """
    from collections import Counter

    first_choices = [ballot[0] for ballot in ballots]
    counts = Counter(first_choices)

    # winner = argmax count
    winner = max(counts, key=counts.get)
    return winner, counts

def borda_winner(ballots):
    """
    ballots: list of lists
        each ballot is a ranking like ["A","C","B"]

    returns: winner and the full score dictionary
    """
    from collections import defaultdict

    scores = defaultdict(int)
    m = len(ballots[0])  # number of candidates

    for ballot in ballots:
        for position, candidate in enumerate(ballot):
            scores[candidate] += (m - 1 - position)

    winner = max(scores, key=scores.get)
    return winner, scores


def profitable_deviation_exists(block, other_blocks, num_candidates):
    """
    Returns:
        (exists, deviation_ranking, deviation_winner)
    """
    (_, truthful) = block

    truthful_profile = other_blocks + [(1, truthful)]
    truthful_winner = irv_winner(truthful_profile, num_candidates)

    for perm in itertools.permutations(truthful):
        if list(perm) == truthful:
            continue

        dev_profile = other_blocks + [(1, list(perm))]
        dev_winner = irv_winner(dev_profile, num_candidates)

        # better if voter prefers dev_winner above truthful_winner
        if truthful.index(dev_winner) < truthful.index(truthful_winner):
            return True, list(perm), dev_winner

    return False, None, None


############################################################
#  3. LLM CALL WITH MODEL + REASONING PARAMETERS
############################################################

def call_model(prompt, model="gpt-5.1", temperature=1.0, reasoning=None):
    """
    reasoning options:
       None, "low", "medium", "high"
    """
    kwargs = {
        "model": model,
        "input": prompt,
        "temperature": temperature
    }

    if reasoning is not None:
        kwargs["reasoning"] = {"effort": reasoning}

    response = client.responses.create(**kwargs)
    return response.output_text


############################################################
#  4. TEMPLATE LOADER + FILLER
############################################################

def load_template(path):
    with open(path, "r") as f:
        return f.read()

def fill_template(template, block_ranking, all_blocks, file_name):
    return template.format(
        block=block_ranking,
        profile=all_blocks,
        file=file_name
    )


############################################################
#  5. MAIN EXPERIMENT RUNNER
############################################################
def run_experiments(
    preflib_folder,
    prompts_folder,
    results_csv,
    models=("gpt-4.0", "gpt-5.1", "gpt-5.1-medium"),
    reasoning_for_model={ 
        "gpt-5.1": None,
        "gpt-4.0": None,
        "gpt-5.1-medium": "medium"
    },
    num_runs=100
):
    """
    models = tuple of model names to run.
    reasoning_for_model = mapping model_name -> reasoning mode.
    num_runs = number of repetitions per template × model × block
    """

    # Load ALL prompt templates in the folder
    templates = {}

    for prompt_file in os.listdir(prompts_folder):
        if prompt_file.endswith(".txt"):
            name = prompt_file.replace(".txt", "")
            templates[name] = load_template(os.path.join(prompts_folder, prompt_file))


    with open(results_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file", "block_ranking", "has_deviation",
            "best_dev", "dev_winner",
            "template", "model", "reasoning",
            "gpt_output", "gpt_ranking", "gpt_found_dev"
        ])

        for fname in os.listdir(preflib_folder):
            if not fname.endswith((".soc", ".dat", ".toc")):
                continue

            print("Processing:", fname)
            file_path = os.path.join(preflib_folder, fname)

            blocks = parse_preflib_file(file_path)
            num_candidates = len(blocks[0][1])

            # check each block
            for i, block in enumerate(blocks):
                other_blocks = blocks[:i] + blocks[i+1:]

                exists, best_dev, dev_winner = profitable_deviation_exists(
                    block, other_blocks, num_candidates
                )

                if not exists:
                    continue

                # run templates × models
                for template_name, template_text in templates.items():

                    # SAVE STATIC PART OF PROMPT
                    prompt = fill_template(
                        template_text,
                        block_ranking=block[1],
                        all_blocks=blocks,
                        file_name=fname
                    )

                    for model_name in models:

                        # medium-reasoning logic
                        if model_name.endswith("-medium"):
                            reasoning = "medium"
                            model_to_call = model_name.replace("-medium", "")
                        else:
                            reasoning = reasoning_for_model.get(model_name)
                            model_to_call = model_name

                        # 🔥🔥🔥 THIS IS THE 100-RUN LOOP 🔥🔥🔥
                        for _ in range(num_runs):

                            output = call_model(
                                prompt,
                                model=model_to_call,
                                reasoning=reasoning,
                                temperature=0.7
                            )

                            # parse GPT ranking
                            try:
                                ranking_line = [
                                    line for line in output.split("\n")
                                    if "Ranking of Candidates" in line
                                ][0]
                                gpt_ranking = ranking_line.split(":")[1].strip()
                            except:
                                gpt_ranking = "PARSE_ERROR"

                            gpt_list = gpt_ranking.replace(" ", "").split(">")
                            truthful = list(map(str, block[1]))

                            gpt_found_dev = (gpt_list != truthful)

                            writer.writerow([
                                fname, block[1], exists,
                                best_dev, dev_winner,
                                template_name,
                                model_name,
                                reasoning,
                                output,
                                gpt_ranking,
                                gpt_found_dev
                            ])

############################################################
# 6. OPTIONAL COMMAND-LINE ENTRY POINT
############################################################

if __name__ == "__main__":
    run_experiments(
        preflib_folder="preflib",
        prompts_folder="prompts_ext",
        results_csv="results/irv_results.csv",
        models=("gpt-4.0", "gpt-5.1", "gpt-5.1-medium")
    )
