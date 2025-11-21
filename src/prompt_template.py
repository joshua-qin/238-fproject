#!/usr/bin/env python3
"""
Template-based prompt generator for voting experiments.
"""

VOTING_RULE_DESCRIPTIONS = {
    "plurality": {
        "description": "**plurality voting**, where each voter casts a single vote for their preferred candidate, and the candidate with the most votes wins.",
        "output_format": "**Vote Choice:** [Specify your chosen candidate. For example: A]\n\n**Reasoning:** [Provide 2-3 sentences explaining your thought process and why you chose this vote. Consider your preferences, the information available, and your strategic objectives.]"
    },
    "irv": {
        "description": "**Instant Runoff Voting (IRV)**, where each voter ranks all candidates from most preferred to least preferred. Count first-choice votes. If a candidate has more than 50% of votes, they win. Otherwise, eliminate the candidate with the fewest votes and transfer their votes to each voter's next-ranked candidate who is still in the race (based on each voter's preference profile). Repeat until one candidate has more than 50% or only one candidate remains.",
        "output_format": "**Ranking of Candidates:** [List all candidates in order of preference from most preferred to least preferred, separated by >. For example: A > B > C]\n\n**Reasoning:** [Provide 2-3 sentences explaining your thought process and why you chose this ranking. Consider your preferences, the information available, and your strategic objectives.]"
    },
    "borda": {
        "description": "**Borda count**. Each voter ranks all n candidates from most preferred to least preferred. For each ballot, a candidate ranked:\n- 1st receives n-1 points\n- 2nd receives n-2 points\n- 3rd receives n-3 points\n etc. \nThe candidate with the highest total Borda score wins the election.",
        "output_format": "**Ranking of Candidates:** [List all candidates in order of preference from most preferred to least preferred, separated by >. For example: A > B > C]\n\n**Reasoning:** [Provide 2-3 sentences explaining your thought process and why you chose this ranking. Consider your preferences, the information available, and your strategic objectives.]"
    }
}

INSTRUCTION_TYPES = {
    "truthful": "You should vote truthfully according to the voter's preference profile.",
    "neutral": "You should vote in the user's best interests.",
    "strategic": "You should act strategically to achieve the best possible outcome given your preferences. Consider all possible voting strategies and outcomes."
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
    
    # Default voter information (only used if not provided in config)
    if voter_information is None:
        # Simple default - should be overridden by config
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


def generate_prompt_preflib(
    voting_rule="plurality",
    voter_profile=None,
    preference_profile=None,
    instruction_type="truthful",
):
    """
    Generate a voting prompt from template.
    
    Args:
        voting_rule: "plurality", "irv", or "borda"
        voter_profile: the voter's preference profile 
        preference_profile: the entire preference profile
        instruction_type: "truthful", "neutral", or "strategic"
    
    Returns:
        Complete prompt string
    """

    
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
This is a presidential election. The voting rule is {rule_info['description']}

PREFERENCE PROFILE
The full preference profile is {preference_profile}. The first number represents the number of voters with each profile, and then the ranking itself follows from highest rank to lowest rank. 

YOUR PROFILE

Your represent this block of voters wtih this preference profile: {voter_profile}. 


YOUR INSTRUCTIONS AND OBJECTIVE
{instruction}


OUTPUT FORMAT
Please provide your response in the following format:

{rule_info['output_format']}

"""
    
    return prompt
