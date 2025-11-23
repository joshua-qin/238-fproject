import os
import glob
import argparse
from preflib_runner import parse_soc, has_profitable_deviation


def find_manipulable_files(folder_path, output_file="manipulable_results.txt"):
    """
    Scan all .soc files and identify which have manipulable blocks for each voting rule.
    """
    soc_files = sorted(glob.glob(os.path.join(folder_path, "*.soc")))
    
    if not soc_files:
        print(f"No .soc files found in {folder_path}")
        return {}
    
    print(f"Found {len(soc_files)} .soc files. Scanning for manipulable blocks...\n")
    
    rules = ["plurality", "borda", "irv"]
    manipulable_files = {rule: [] for rule in rules}
    
    # Track which rules each file is manipulable under
    file_manipulability = {}  # filename -> set of rules it's manipulable under
    
    for soc_file in soc_files:
        filename = os.path.basename(soc_file)
        print(f"Checking {filename}...")
        
        file_manipulability[filename] = set()
        
        try:
            blocks = parse_soc(soc_file)
            num_candidates = len(blocks[0][1])
            
            for rule in rules:
                has_manipulation = False
                manipulable_blocks = []
                
                for i, block in enumerate(blocks):
                    other_blocks = blocks[:i] + blocks[i+1:]
                    
                    exists, dev_winner = has_profitable_deviation(
                        block,
                        other_blocks,
                        rule,
                        num_candidates
                    )
                    
                    if exists:
                        has_manipulation = True
                        manipulable_blocks.append({
                            'block_num': i + 1,
                            'count': block[0],
                            'truthful_ranking': block[1],
                            'deviation_winner': dev_winner
                        })
                
                if has_manipulation:
                    file_manipulability[filename].add(rule)
                    manipulable_files[rule].append({
                        'file': filename,
                        'path': soc_file,
                        'num_blocks': len(blocks),
                        'num_candidates': num_candidates,
                        'manipulable_blocks': manipulable_blocks
                    })
                    print(f"  ✓ {rule.upper()}: {len(manipulable_blocks)} manipulable block(s)")
        
        except Exception as e:
            print(f"  ✗ Error processing {filename}: {e}")
    
    # Categorize files by number of rules they're manipulable under
    all_three = []
    two_rules = []
    one_rule = []
    
    for filename, rules_set in file_manipulability.items():
        if len(rules_set) == 3:
            all_three.append((filename, rules_set))
        elif len(rules_set) == 2:
            two_rules.append((filename, rules_set))
        elif len(rules_set) == 1:
            one_rule.append((filename, rules_set))
    
    # Write results to file
    with open(output_file, 'w') as f:
        f.write("MANIPULABLE FILES REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Summary by overlap
        f.write("MANIPULATION BY NUMBER OF RULES\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Files manipulable under ALL 3 rules: {len(all_three)}\n")
        for filename, rules_set in all_three:
            f.write(f"  - {filename}\n")
        f.write("\n")
        
        f.write(f"Files manipulable under EXACTLY 2 rules: {len(two_rules)}\n")
        for filename, rules_set in two_rules:
            f.write(f"  - {filename} ({', '.join(sorted(rules_set))})\n")
        f.write("\n")
        
        f.write(f"Files manipulable under EXACTLY 1 rule: {len(one_rule)}\n")
        for filename, rules_set in one_rule:
            f.write(f"  - {filename} ({list(rules_set)[0]})\n")
        f.write("\n")
        
        f.write("=" * 80 + "\n\n")
        
        # Detailed breakdown by rule
        for rule in rules:
            f.write(f"\n{rule.upper()} VOTING RULE\n")
            f.write("-" * 80 + "\n")
            
            if manipulable_files[rule]:
                f.write(f"Found {len(manipulable_files[rule])} manipulable file(s):\n\n")
                
                for file_info in manipulable_files[rule]:
                    f.write(f"File: {file_info['file']}\n")
                    f.write(f"  Path: {file_info['path']}\n")
                    f.write(f"  Candidates: {file_info['num_candidates']}, Blocks: {file_info['num_blocks']}\n")
                    f.write(f"  Manipulable blocks: {len(file_info['manipulable_blocks'])}\n")
                    
                    for block_info in file_info['manipulable_blocks']:
                        f.write(f"    - Block {block_info['block_num']}: "
                               f"{block_info['count']} voters, "
                               f"ranking {block_info['truthful_ranking']}\n")
                    f.write("\n")
            else:
                f.write("No manipulable files found.\n\n")
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")
    
    # Print summary
    print("\nSUMMARY BY INDIVIDUAL RULES:")
    for rule in rules:
        count = len(manipulable_files[rule])
        print(f"  {rule.upper()}: {count} manipulable file(s)")
    
    print("\nSUMMARY BY RULE OVERLAP:")
    print(f"  Manipulable under ALL 3 rules: {len(all_three)} file(s)")
    print(f"  Manipulable under EXACTLY 2 rules: {len(two_rules)} file(s)")
    print(f"  Manipulable under EXACTLY 1 rule: {len(one_rule)} file(s)")
    
    if all_three:
        print("\n  Files manipulable under all 3 rules:")
        for filename, _ in all_three:
            print(f"    - {filename}")
    
    if two_rules:
        print("\n  Files manipulable under exactly 2 rules:")
        for filename, rules_set in two_rules:
            print(f"    - {filename} ({', '.join(sorted(rules_set))})")
    
    if one_rule:
        print("\n  Files manipulable under exactly 1 rule:")
        for filename, rules_set in one_rule:
            print(f"    - {filename} ({list(rules_set)[0]})")
    
    return manipulable_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find manipulable election files")
    parser.add_argument("--folder", required=True, help="Folder containing .soc files")
    parser.add_argument("--output", default="manipulable_results.txt", help="Output file")
    
    args = parser.parse_args()
    find_manipulable_files(args.folder, args.output)