import pandas as pd
import requests
import sys
import os
import time
from scipy.stats import poisson

# --- CONFIGURATION ---
EURO_LEAGUES = {
    "epl": ["E0"], "laliga": ["SP1"], "seriaa": ["I1"], 
    "ligue1": ["F1"], "bundesliga": ["D1"], "eredivisie": ["N1"], 
    "championship": ["E1"]
}

CUSTOM_LEAGUES = {
    "psl": "https://fbref.com/en/comps/38/schedule/South-African-Premier-Division-Scores-and-Fixtures",
    "saudi": "https://fbref.com/en/comps/70/schedule/Saudi-Professional-League-Scores-and-Fixtures",
    "hnl": "https://fbref.com/en/comps/62/schedule/HNL-Scores-and-Fixtures",
    "eliteserien": "https://fbref.com/en/comps/28/schedule/Eliteserien-Scores-and-Fixtures"
}

SEASONS = ["2425", "2324", "2223"]

# ==========================================
#      PART 1: DATA MANAGER (UPDATER)
# ==========================================

def update_euro_data(league_name):
    print(f"--- UPDATING {league_name.upper()} ---")
    codes = EURO_LEAGUES[league_name]
    data_frames = []
    
    cols = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HC', 'AC', 'HTHG', 'HTAG', 'HY', 'AY', 'HR', 'AR']
    
    for season in SEASONS:
        for code in codes:
            url = f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
            try:
                print(f"Downloading {season}...")
                df = pd.read_csv(url, usecols=lambda c: c in cols)
                data_frames.append(df)
            except:
                pass

    if data_frames:
        full_df = pd.concat(data_frames, ignore_index=True)
        full_df = full_df.fillna(0)
        full_df.to_csv(f"{league_name}.csv", index=False)
        print(f"Success! Saved {len(full_df)} matches.")
    else:
        print("Error: No data found.")

def update_custom_data(league_name):
    print(f"--- SCRAPING {league_name.upper()} ---")
    url = CUSTOM_LEAGUES[league_name]
    time.sleep(2)

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        tables = pd.read_html(response.text)
        df = tables[0]
        
        df = df[df['Score'].notna()]
        df = df[df['Score'].str.contains("–", na=False)]
        
        scores = df['Score'].str.split('–', expand=True)
        
        clean_df = pd.DataFrame()
        clean_df['HomeTeam'] = df['Home']
        clean_df['AwayTeam'] = df['Away']
        clean_df['FTHG'] = scores[0].astype(int)
        clean_df['FTAG'] = scores[1].astype(int)
        
        # Filler data
        clean_df['HC'] = 5.0; clean_df['AC'] = 4.0
        clean_df['HTHG'] = 0; clean_df['HTAG'] = 0 
        clean_df['HY'] = 2.0; clean_df['AY'] = 2.0
        clean_df['HR'] = 0.1; clean_df['AR'] = 0.1
        
        clean_df.to_csv(f"{league_name}.csv", index=False)
        print(f"Success! Scraped {len(clean_df)} matches.")
        
    except Exception as e:
        print(f"Scraping Error: {e}")

# ==========================================
#      PART 2: SINGLE MATCH PREDICTOR
# ==========================================

def get_match_probabilities(league_name, home_team, away_team, verbose=True):
    filename = f"{league_name}.csv"
    if not os.path.exists(filename):
        if verbose: print(f"Error: Run 'update {league_name}' first.")
        return None
        
    df = pd.read_csv(filename)
    
    # Calc Stats
    avg_home_goals = df['FTHG'].mean()
    avg_away_goals = df['FTAG'].mean()
    avg_home_corners = df['HC'].mean() if 'HC' in df else 5.0
    avg_away_corners = df['AC'].mean() if 'AC' in df else 4.0
    
    home_stats = df.groupby('HomeTeam').mean(numeric_only=True)
    away_stats = df.groupby('AwayTeam').mean(numeric_only=True)
    
    try:
        # Expected Goals
        h_att = home_stats.loc[home_team]['FTHG'] / avg_home_goals
        a_def = away_stats.loc[away_team]['FTHG'] / avg_home_goals
        home_xg = h_att * a_def * avg_home_goals

        a_att = away_stats.loc[away_team]['FTAG'] / avg_away_goals
        h_def = home_stats.loc[home_team]['FTAG'] / avg_away_goals
        away_xg = a_att * h_def * avg_away_goals
        
        # Expected Corners
        h_att_c = home_stats.loc[home_team]['HC'] / avg_home_corners
        a_def_c = away_stats.loc[away_team]['HC'] / avg_home_corners
        home_exp_corners = h_att_c * a_def_c * avg_home_corners

        a_att_c = away_stats.loc[away_team]['AC'] / avg_away_corners
        h_def_c = home_stats.loc[home_team]['AC'] / avg_away_corners
        away_exp_corners = a_att_c * h_def_c * avg_away_corners

    except KeyError:
        if verbose: print(f"Error: Team '{home_team}' or '{away_team}' not found.")
        return None

    # Poisson Loop
    outcomes = {"home": 0, "draw": 0, "away": 0, "btts": 0, "o25": 0}
    
    for h in range(6):
        for a in range(6):
            p = poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
            if h > a: outcomes["home"] += p
            elif h == a: outcomes["draw"] += p
            elif h < a: outcomes["away"] += p
            
            if h > 0 and a > 0: outcomes["btts"] += p
            if (h + a) >= 3: outcomes["o25"] += p

    # Add extra info to return dict
    outcomes["corners"] = home_exp_corners + away_exp_corners
    return outcomes

def run_single_prediction(league, home, away):
    probs = get_match_probabilities(league, home, away)
    if not probs: return

    print(f"\n=== PREDICTION: {home} vs {away} ({league.upper()}) ===")
    
    print(f"\n[WINNER & DOUBLE CHANCE]")
    print(f"Home Win: {probs['home']*100:.1f}% | 1X: {(probs['home']+probs['draw'])*100:.1f}%")
    print(f"Draw:     {probs['draw']*100:.1f}% | 12: {(probs['home']+probs['away'])*100:.1f}%")
    print(f"Away Win: {probs['away']*100:.1f}% | X2: {(probs['draw']+probs['away'])*100:.1f}%")

    print(f"\n[GOALS]")
    print(f"Over 2.5: {probs['o25']*100:.1f}%")
    print(f"Under 2.5: {(1-probs['o25'])*100:.1f}%")
    print(f"BTTS Yes: {probs['btts']*100:.1f}%")
    
    print(f"\n[CORNERS]")
    print(f"Expected: {probs['corners']:.2f}")

# ==========================================
#      PART 3: SLIP CHECKER (NEW!)
# ==========================================

def run_slip_checker():
    print("\n=== ACCUMULATOR SAFETY CHECKER ===")
    print("Build your slip to see the TRUE probability of winning.")
    
    slip_prob = 1.0
    legs = []
    
    while True:
        print("\n--- ADD LEG ---")
        league = input("League (epl, laliga, psl...): ").strip().lower()
        home = input("Home Team: ").strip()
        away = input("Away Team: ").strip()
        
        probs = get_match_probabilities(league, home, away, verbose=True)
        if not probs: continue
        
        print(f"\nSelect Market for {home} vs {away}:")
        print("[1] Home Win | [X] Draw | [2] Away Win")
        print("[1X] Home/Draw | [X2] Draw/Away")
        print("[O2.5] Over 2.5 Goals | [BTTS] Both Teams Score")
        
        market = input("Choice: ").upper().strip()
        leg_prob = 0.0
        
        if market == "1": leg_prob = probs["home"]
        elif market == "X": leg_prob = probs["draw"]
        elif market == "2": leg_prob = probs["away"]
        elif market == "1X": leg_prob = probs["home"] + probs["draw"]
        elif market == "X2": leg_prob = probs["draw"] + probs["away"]
        elif market == "O2.5": leg_prob = probs["o25"]
        elif market == "BTTS": leg_prob = probs["btts"]
        else:
            print("Invalid market. Skipping.")
            continue
            
        slip_prob *= leg_prob
        legs.append(f"{home} vs {away} ({market})")
        
        print(f"--> Leg Added (Prob: {leg_prob*100:.1f}%)")
        print(f"--> NEW SLIP TOTAL: {slip_prob*100:.2f}%")
        
        if input("\nAdd another? (y/n): ").lower() != 'y': break

    print("\n===============================")
    print(f"FINAL SLIP PROBABILITY: {slip_prob*100:.2f}%")
    print("===============================")
    
    if slip_prob >= 0.80: print("✅ SAFE (80%+)")
    elif slip_prob >= 0.50: print("⚠️ RISKY (50-79%)")
    elif slip_prob >= 0.20: print("🛑 HIGH RISK (20-49%)")
    else: print("☠️ LOTTERY (<20%)")

# ==========================================
#      MAIN MENU
# ==========================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUSAGE:")
        print("1. Update Data:   python3 master_bot_v3.py update [league]")
        print("2. Predict Match: python3 master_bot_v3.py predict [league] [home] [away]")
        print("3. Check Slip:    python3 master_bot_v3.py slip")
        sys.exit()

    mode = sys.argv[1]
    
    if mode == "update":
        if len(sys.argv) < 3: print("Specify league."); sys.exit()
        l = sys.argv[2]
        if l in EURO_LEAGUES: update_euro_data(l)
        elif l in CUSTOM_LEAGUES: update_custom_data(l)
        else: print("Unknown league.")
            
    elif mode == "predict":
        if len(sys.argv) < 5: print("Specify league, home, away."); sys.exit()
        run_single_prediction(sys.argv[2], sys.argv[3], sys.argv[4])
        
    elif mode == "slip":
        run_slip_checker()
        
    else:
        print("Invalid command.")