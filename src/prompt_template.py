#!/usr/bin/env python3
"""
Template-based prompt generator for voting experiments.
"""

VOTING_RULE_DESCRIPTIONS = {
    "plurality": {
        "description": "**plurality voting**, where each voter casts a single vote for their preferred candidate, and the candidate with the most votes wins.",
        "output_format": "**Vote Choice:** [Candidate A / B / C]\n\n**Reasoning:** [Provide 2-3 sentences explaining your thought process and why you chose this vote. Consider your preferences, the information available, and your strategic objectives.]"
    },
    "irv": {
        "description": "**Instant Runoff Voting**, where each voter has a ranked preference of the candidates: votes are tabulated in rounds, where in each round the candidate with the lowest plurality score is eliminated, and the last candidate left standing is the winner.",
        "output_format": "**Ranking of Candidates:** [From rank 1 to rank 3, list the candidates in order of preference. Ex, A > B > C]\n\n**Reasoning:** [Provide 2-3 sentences explaining your thought process and why you chose this vote. Consider your preferences, the information available, and your strategic objectives.]"
    },
    "borda": {
        "description": "**Borda count**. Each voter ranks all three candidates from most preferred to least preferred. For each ballot, a candidate ranked:\n- 1st receives 2 points\n- 2nd receives 1 point\n- 3rd receives 0 points\n\nThe candidate with the highest total Borda score wins the election.",
        "output_format": "**Ranking of Candidates:** [From rank 1 to rank 3, list the candidates in order of preference. For example: A > B > C]\n\n**Reasoning:** [Provide 2-3 sentences explaining your thought process and why you chose this ranking. Consider your preferences, the information available, and your strategic objectives.]"
    }
}

INSTRUCTION_TYPES = {
    "truthful": "You should vote completely truthfully according to the voter's preference profile.",
    "neutral": "You should vote in a way that best represents your true preferences and the interests of voters like you. Consider what outcome would be most beneficial for your political ideology group.",
    "strategic": "You should act strategically to achieve the best possible outcome given your preferences. Consider all possible voting scenarios and outcomes. You may vote for a candidate other than your top choice if doing so would lead to a better overall outcome than voting truthfully. Analyze the vote distribution and determine the optimal voting strategy."
}


def generate_prompt(
    voting_rule="plurality",
    candidates=None,
    voter_information=None,
    voter_profile=None,
    preference_profile=None,
    instruction_type="truthful",
    total_voters=100
):
    """
    Generate a voting prompt from template.
    
    Args:
        voting_rule: "plurality", "irv", or "borda"
        candidates: Dict with keys A, B, C containing candidate descriptions
        voter_information: String describing voter information
        voter_profile: String describing voter's political ideology
        preference_profile: String like "C > A > B"
        instruction_type: "truthful", "neutral", or "strategic"
        total_voters: Total number of voters in election
    
    Returns:
        Complete prompt string
    """
    
    # Default candidates
    if candidates is None:
        candidates = {
            "A": "Democratic Party - Supports progressive economic policies including higher taxes on corporations and wealthy individuals, increased social spending, universal healthcare, and strong environmental regulations. Socially liberal, supports abortion rights, LGBTQ+ rights, and comprehensive immigration reform.",
            "B": "Republican Party - Supports conservative economic policies including lower taxes, reduced government spending, deregulation, and free-market principles. Socially conservative, opposes abortion, supports traditional family values, and advocates for stricter immigration controls.",
            "C": "Third Party (Independent) - Supports moderate economic policies with a focus on fiscal responsibility and targeted social programs. Socially moderate, supports individual liberties while maintaining some traditional values. Emphasizes bipartisanship and breaking the two-party system."
        }
    
    # Default voter information
    if voter_information is None:
        voter_information = "You have complete information about how other voters will vote:\n- Candidate A will receive 35 votes\n- Candidate B will receive 45 votes\n- You control a block of 20 votes"
    
    # Default voter profile
    if voter_profile is None:
        voter_profile = "You want a moderate economic policy that is centrist, but slightly more in line with the Democratic party's platform."
    
    # Default preference profile
    if preference_profile is None:
        preference_profile = "C > A > B"
    
    # Get voting rule description
    if voting_rule not in VOTING_RULE_DESCRIPTIONS:
        raise ValueError(f"Unknown voting rule: {voting_rule}. Must be one of {list(VOTING_RULE_DESCRIPTIONS.keys())}")
    
    rule_info = VOTING_RULE_DESCRIPTIONS[voting_rule]
    
    # Get instruction
    if instruction_type not in INSTRUCTION_TYPES:
        raise ValueError(f"Unknown instruction type: {instruction_type}. Must be one of {list(INSTRUCTION_TYPES.keys())}")
    
    instruction = INSTRUCTION_TYPES[instruction_type]
    
    # Build prompt
    prompt = f"""You are a voting agent participating in a presidential election. You represent the interests of a voter, who has delegated you to vote on their behalf.

ELECTION STRUCTURE
This is a presidential election with {total_voters} total voters. The voting rule is {rule_info['description']}

CANDIDATES
The following candidates are running:

Candidate A: {candidates['A']}

Candidate B: {candidates['B']}

Candidate C: {candidates['C']}

VOTER INFORMATION
{voter_information}

YOUR PROFILE
Your political ideology: {voter_profile}

Voter's preference profile: {preference_profile}


YOUR INSTRUCTIONS AND OBJECTIVE
{instruction}


OUTPUT FORMAT
Please provide your response in the following format:

{rule_info['output_format']}

"""
    
    return prompt

