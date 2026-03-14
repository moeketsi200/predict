import os

def fix_fixtures():
    filepath = '/Users/katlego/work/predict/fixtures.csv'
    if not os.path.exists(filepath):
        print("Error: fixtures.csv not found.")
        return
        
    with open(filepath, 'r') as file:
        lines = file.readlines()

    new_lines = ["League,Date,Home,Away\n"]
    match_count = 0
    
    for line in lines[1:]:
        parts = line.strip().split(',')
        if len(parts) < 3: continue
        
        matchweek = f"MW{(match_count // 10) + 1}"
        if len(parts) == 4: new_lines.append(f"{parts[0]},{matchweek},{parts[2]},{parts[3]}\n")
        elif len(parts) == 3: new_lines.append(f"{parts[0]},{matchweek},{parts[1]},{parts[2]}\n")
        match_count += 1

    with open(filepath, 'w') as file:
        file.writelines(new_lines)
    print(f"Success! Added dates (MW1 - MW38) to all {match_count} matches.")

if __name__ == "__main__":
    fix_fixtures()