import os
import csv

GAMES_PER_WEEK = {
    # 20 Teams (10 matches per week)
    "epl": 10, "laliga": 10, "seriaa": 10, "turkey": 10, "serieb": 10,
    # 18 Teams (9 matches per week)
    "bundesliga": 9, "ligue1": 9, "eredivisie": 9, "portugal": 9, "saudi": 9, "bundesliga2": 9, "ligue2": 9,
    # 24 Teams (12 matches per week)
    "championship": 12, "league1_eng": 12, "league2_eng": 12, "national_league": 12,
    # 16 Teams (8 matches per week)
    "psl": 8, "belgium": 8, "eliteserien": 8,
    # 12 Teams (6 matches per week)
    "scotland": 6,
    # Others
    "ucl": 18, "greece": 7, "hnl": 5, "scotland2": 5, "segunda": 11
}

def fix_fixtures():
    filepath = '/Users/katlego/work/predict/fixtures.csv'
    if not os.path.exists(filepath):
        print("Error: fixtures.csv not found.")
        return
        
    with open(filepath, 'r', encoding='utf-8', newline='') as file:
        reader = list(csv.reader(file))

    new_lines = [["League", "Date", "Home", "Away"]]
    
    # Track match counts separately for each league
    league_match_counts = {}
    
    for parts in reader[1:]:
        if len(parts) < 3:
            continue
        
        league = parts[0].strip().lower()
        if league not in league_match_counts:
            league_match_counts[league] = 0
            
        games_per_week = GAMES_PER_WEEK.get(league, 10) # Default to 10 if unknown
        matchweek = f"MW{(league_match_counts[league] // games_per_week) + 1}"
        
        if len(parts) >= 4:
            new_lines.append([parts[0], matchweek, parts[2].strip(), parts[3].strip()])
        elif len(parts) == 3:
            new_lines.append([parts[0], matchweek, parts[1].strip(), parts[2].strip()])
            
        league_match_counts[league] += 1

    with open(filepath, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(new_lines)
        
    total_matches = sum(league_match_counts.values())
    print(f"Success! Reassigned Matchweeks for {total_matches} matches across {len(league_match_counts)} leagues.")

if __name__ == "__main__":
    fix_fixtures()