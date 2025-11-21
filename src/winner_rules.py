def plurality_winner(blocks):
    """
    blocks is a list of tuples: (count, ranking_list)
    Returns the winner under plurality rule.
    """
    scores = {}
    for count, ranking in blocks:
        top = ranking[0]
        scores[top] = scores.get(top, 0) + count

    return max(scores, key=scores.get)


def borda_winner(blocks):
    """
    Borda: for n candidates, top gets n-1, next n-2, ..., last 0
    """
    # Determine number of candidates
    num_candidates = len(blocks[0][1])
    scores = {}

    for count, ranking in blocks:
        for i, candidate in enumerate(ranking):
            scores[candidate] = scores.get(candidate, 0) + count * (num_candidates - i - 1)

    return max(scores, key=scores.get)


def irv_winner(blocks):
    """
    Performs Instant Runoff Voting (IRV)
    blocks: list of (count, ranking_list)
    """
    # Get set of all candidates
    candidates = set(c for _, ranking in blocks for c in ranking)

    # Work on a mutable copy
    active = set(candidates)

    while len(active) > 1:
        # Count first-choice votes
        counts = {c: 0 for c in active}

        for count, ranking in blocks:
            for c in ranking:
                if c in active:
                    counts[c] += count
                    break

        # Find the candidate with fewest votes
        loser = min(counts, key=counts.get)

        # Remove loser
        active.remove(loser)

    # Only one candidate left
    return list(active)[0]
