import pandas as pd
import requests
import os
import time
import argparse
import csv
import numpy as np
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

FBREF_LEAGUE_IDS = {
    "epl": "9", "laliga": "12", "seriaa": "11", "ligue1": "13", 
    "bundesliga": "20", "eredivisie": "21", "championship": "10",
    "psl": "38", "saudi": "70", "hnl": "62", "eliteserien": "28"
}

SEASONS = ["2425", "2324", "2223"]

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Auto-migrate existing database files to keep the root folder clean
for f in os.listdir('.'):
    if f.endswith('.csv') and f not in ['fixtures.csv', 'sim_results.csv']:
        try:
            os.rename(f, os.path.join(DATA_DIR, f))
        except Exception:
            pass

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
            except Exception as e:
                print(f"Warning: Could not download data for {league_name} {season} from {url}. Error: {e}")

    if data_frames:
        full_df = pd.concat(data_frames, ignore_index=True)
        full_df = full_df.fillna(0)
        full_df.to_csv(os.path.join(DATA_DIR, f"{league_name}.csv"), index=False)
        print(f"Success! Saved {len(full_df)} matches.")
    else:
        print("Error: No data found.")

def update_custom_data(league_name):
    print(f"--- SCRAPING {league_name.upper()} ---")
    url = CUSTOM_LEAGUES[league_name]
    time.sleep(2)

    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
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

        # Note: fbref scraping currently only supports full-time goals.
        # Other stats (corners, cards, HT goals) are not available.
        clean_df.to_csv(os.path.join(DATA_DIR, f"{league_name}.csv"), index=False)
        print(f"Success! Scraped {len(clean_df)} matches.")
        
    except Exception as e:
        print(f"Scraping Error: {e}")

# ==========================================
#      PART 2: SINGLE MATCH PREDICTOR
# ==========================================

def get_match_probabilities(league_name, home_team, away_team, verbose=True):
    filename = os.path.join(DATA_DIR, f"{league_name}.csv")
    if not os.path.exists(filename):
        if verbose: print(f"Error: Run 'update {league_name}' first.")
        return None
        
    df = pd.read_csv(filename)
    df = df.fillna(0)
    if len(df) == 0: return None
    
    # --- TIME-WEIGHTED FORM (Exponential Smoothing) ---
    # Older matches get ~0.36 weight, newest matches get 1.0 weight
    df['Weight'] = np.exp(np.linspace(-1, 0, len(df)))
    
    avg_home_goals = np.average(df['FTHG'], weights=df['Weight'])
    avg_away_goals = np.average(df['FTAG'], weights=df['Weight'])
    
    def weighted_mean(group, col):
        w_sum = group['Weight'].sum()
        return (group[col] * group['Weight']).sum() / w_sum if w_sum > 0 else 0

    home_groups = df.groupby('HomeTeam')
    away_groups = df.groupby('AwayTeam')
    
    try:
        # --- GRACEFUL FALLBACKS FOR MISSING TEAMS ---
        if home_team in home_groups.groups:
            h_fthg = weighted_mean(home_groups.get_group(home_team), 'FTHG')
            h_ftag = weighted_mean(home_groups.get_group(home_team), 'FTAG')
        else:
            if verbose: print(f"Note: No data for '{home_team}'. Using league averages.")
            h_fthg, h_ftag = avg_home_goals, avg_away_goals

        if away_team in away_groups.groups:
            a_ftag = weighted_mean(away_groups.get_group(away_team), 'FTAG')
            a_fthg = weighted_mean(away_groups.get_group(away_team), 'FTHG')
        else:
            if verbose: print(f"Note: No data for '{away_team}'. Using league averages.")
            a_ftag, a_fthg = avg_away_goals, avg_home_goals

        h_att = h_fthg / avg_home_goals if avg_home_goals > 0 else 1.0
        h_def = h_ftag / avg_away_goals if avg_away_goals > 0 else 1.0
        a_att = a_ftag / avg_away_goals if avg_away_goals > 0 else 1.0
        a_def = a_fthg / avg_home_goals if avg_home_goals > 0 else 1.0

        home_xg = h_att * a_def * avg_home_goals
        away_xg = a_att * h_def * avg_away_goals
        
        # Expected Corners (only if data is available)
        home_exp_corners = None
        away_exp_corners = None
        
        # Filter out rows with no corner data so 0s don't ruin the averages
        if 'HC' in df.columns and 'AC' in df.columns:
            df_corners = df[(df['HC'] > 0) | (df['AC'] > 0)]
            if len(df_corners) > 0:
                avg_hc = np.average(df_corners['HC'], weights=df_corners['Weight'])
                avg_ac = np.average(df_corners['AC'], weights=df_corners['Weight'])

                home_groups_c = df_corners.groupby('HomeTeam')
                away_groups_c = df_corners.groupby('AwayTeam')

                h_hc = weighted_mean(home_groups_c.get_group(home_team), 'HC') if home_team in home_groups_c.groups else avg_hc
                h_ac = weighted_mean(home_groups_c.get_group(home_team), 'AC') if home_team in home_groups_c.groups else avg_ac
                a_hc = weighted_mean(away_groups_c.get_group(away_team), 'HC') if away_team in away_groups_c.groups else avg_hc
                a_ac = weighted_mean(away_groups_c.get_group(away_team), 'AC') if away_team in away_groups_c.groups else avg_ac

                h_att_c = h_hc / avg_hc if avg_hc > 0 else 1.0
                a_def_c = a_hc / avg_hc if avg_hc > 0 else 1.0
                home_exp_corners = h_att_c * a_def_c * avg_hc

                a_att_c = a_ac / avg_ac if avg_ac > 0 else 1.0
                h_def_c = h_ac / avg_ac if avg_ac > 0 else 1.0
                away_exp_corners = a_att_c * h_def_c * avg_ac
    except KeyError:
        if verbose: print(f"Error: Team '{home_team}' or '{away_team}' not found.")
        return None

    # Poisson Loop
    outcomes = {"home": 0, "draw": 0, "away": 0, "btts": 0, "o15": 0, "o25": 0, "h_minus_1_5": 0, "a_minus_1_5": 0}
    rho = 0.15 # Dixon-Coles correlation parameter for low-scoring matches
    total_prob = 0
    
    for h in range(7):
        for a in range(7):
            p = poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
            
            # --- DIXON-COLES ADJUSTMENT ---
            if h == 0 and a == 0: p *= max(0, 1 - rho * home_xg * away_xg)
            elif h == 1 and a == 0: p *= max(0, 1 + rho * away_xg)
            elif h == 0 and a == 1: p *= max(0, 1 + rho * home_xg)
            elif h == 1 and a == 1: p *= max(0, 1 - rho)
            
            total_prob += p
            
            if h > a: outcomes["home"] += p
            elif h == a: outcomes["draw"] += p
            elif h < a: outcomes["away"] += p
            
            if h > 0 and a > 0: outcomes["btts"] += p
            if (h + a) >= 2: outcomes["o15"] += p
            if (h + a) >= 3: outcomes["o25"] += p

            if h - a >= 2: outcomes["h_minus_1_5"] += p
            if a - h >= 2: outcomes["a_minus_1_5"] += p

    # Normalize probabilities
    for key in outcomes:
        outcomes[key] /= total_prob
        
    # --- DRAW NO BET (DNB) ---
    win_total = outcomes["home"] + outcomes["away"]
    if win_total > 0:
        outcomes["dnb_home"] = outcomes["home"] / win_total
        outcomes["dnb_away"] = outcomes["away"] / win_total
    else:
        outcomes["dnb_home"], outcomes["dnb_away"] = 0.0, 0.0

    # Add extra info to return dict
    if home_exp_corners is not None and away_exp_corners is not None:
        outcomes["corners"] = home_exp_corners + away_exp_corners

    return outcomes

def run_single_prediction(league, home, away):
    probs = get_match_probabilities(league, home, away)
    if not probs: return

    print(f"\n=== PREDICTION: {home} vs {away} ({league.upper()}) ===")
    
    print(f"\n[WINNER, DOUBLE CHANCE & DNB]")
    print(f"Home: {probs['home']*100:.1f}% | 1X: {(probs['home']+probs['draw'])*100:.1f}% | DNB: {probs['dnb_home']*100:.1f}%")
    print(f"Draw: {probs['draw']*100:.1f}% | 12: {(probs['home']+probs['away'])*100:.1f}%")
    print(f"Away: {probs['away']*100:.1f}% | X2: {(probs['draw']+probs['away'])*100:.1f}% | DNB: {probs['dnb_away']*100:.1f}%")

    print(f"\n[GOALS]")
    print(f"Over 1.5: {probs['o15']*100:.1f}% | Over 2.5: {probs['o25']*100:.1f}%")
    print(f"Under 2.5: {(1-probs['o25'])*100:.1f}%")
    print(f"BTTS Yes: {probs['btts']*100:.1f}%")
    
    print(f"\n[HANDICAP (-1.5)]")
    print(f"Home (-1.5): {probs['h_minus_1_5']*100:.1f}% (Win by 2+ goals)")
    print(f"Away (-1.5): {probs['a_minus_1_5']*100:.1f}% (Win by 2+ goals)")

    if "corners" in probs:
        print(f"  [CORNERS]")
        print(f"Expected: {probs['corners']:.2f}")

# ==========================================
#      PART 3: SLIP CHECKER (NEW!)
# ==========================================

def run_slip_checker():
    print("\n=== ACCUMULATOR SAFETY CHECKER ===")
    print("Build your slip to see the TRUE probability of winning.")
    
    slip_prob = 1.0
    legs = []
    
    MARKET_PROB_MAP = {
        "1":    lambda p: p["home"],
        "X":    lambda p: p["draw"],
        "2":    lambda p: p["away"],
        "1X":   lambda p: p["home"] + p["draw"],
        "X2":   lambda p: p["draw"] + p["away"],
        "DNB1": lambda p: p["dnb_home"],
        "DNB2": lambda p: p["dnb_away"],
        "O1.5": lambda p: p["o15"],
        "O2.5": lambda p: p["o25"],
        "BTTS": lambda p: p["btts"],
        "H-1.5": lambda p: p["h_minus_1_5"],
        "A-1.5": lambda p: p["a_minus_1_5"],
    }

    while True:
        print("\n--- ADD LEG ---")
        league = input("League (epl, laliga, psl...): ").strip().lower()
        home = input("Home Team: ").strip()
        away = input("Away Team: ").strip()
        
        probs = get_match_probabilities(league, home, away, verbose=True)
        if not probs: continue
        
        print(f"\nSelect Market for {home} vs {away}:")
        print("[1] Home Win | [X] Draw | [2] Away Win")
        print("[1X] Home/Draw | [X2] Draw/Away | [DNB1] Home DNB | [DNB2] Away DNB")
        print("[O1.5] Over 1.5 | [O2.5] Over 2.5 | [BTTS] Both Teams Score")
        print("[H-1.5] Home Win by 2+ | [A-1.5] Away Win by 2+")
        
        market = input("Choice: ").upper().strip()
        
        if market not in MARKET_PROB_MAP:
            print("Invalid market. Skipping.")
            continue
        
        leg_prob = MARKET_PROB_MAP[market](probs)
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

def run_batch_predictions(filepath, date_filter=None):
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        return

    try:
        df = pd.read_csv(filepath)
        required_cols = ['League', 'Home', 'Away']
        if not all(col in df.columns for col in required_cols):
            print(f"Error: CSV must contain columns: {', '.join(required_cols)}")
            return
            
        if date_filter:
            if 'Date' not in df.columns:
                print("Error: To filter by date, your CSV must have a 'Date' column.")
                return
            df = df[df['Date'].astype(str).str.strip().str.upper() == str(date_filter).strip().upper()]
            if df.empty:
                print(f"No matches found for date/week: {date_filter}")
                return
        
        print(f"\n=== BATCH PREDICTIONS FOR {len(df)} MATCHES ===")
        for index, row in df.iterrows():
            league = str(row['League']).strip().lower()
            home = str(row['Home']).strip()
            away = str(row['Away']).strip()
            
            probs = get_match_probabilities(league, home, away, verbose=False)
            if not probs:
                print(f"\n[ {league.upper()} ] {home} vs {away} ❌ Data or teams not found. Skipping.")
                continue
            
            markets = {
                "Home Win (1)": probs['home'],
                "Away Win (2)": probs['away'],
                "Home DNB (DNB1)": probs['dnb_home'],
                "Away DNB (DNB2)": probs['dnb_away'],
                "Double Chance (1X)": probs['home'] + probs['draw'],
                "Double Chance (X2)": probs['away'] + probs['draw'],
                "Over 1.5 Goals": probs['o15'],
                "Over 2.5 Goals": probs['o25'],
                "Under 2.5 Goals": 1 - probs['o25'],
                "BTTS (Yes)": probs['btts'],
                "Home -1.5": probs['h_minus_1_5'],
                "Away -1.5": probs['a_minus_1_5']
            }
            safest_market = max(markets, key=markets.get)
            safest_prob = markets[safest_market] * 100
            
            print(f"\n[ {league.upper()} ] {home} vs {away}")
            print(f"  Win: 1 ({probs['home']*100:.1f}%) | X ({probs['draw']*100:.1f}%) | 2 ({probs['away']*100:.1f}%) | DNB1 ({probs['dnb_home']*100:.1f}%)")
            print(f"  Handicap (-1.5): Home ({probs['h_minus_1_5']*100:.1f}%) | Away ({probs['a_minus_1_5']*100:.1f}%)")
            print(f"  Goals: O1.5 ({probs['o15']*100:.1f}%) | O2.5 ({probs['o25']*100:.1f}%) | BTTS ({probs['btts']*100:.1f}%)")
            if "corners" in probs: print(f"  Corners: {probs['corners']:.1f}")
            print(f"  ⭐ Safest Bet: {safest_market} ({safest_prob:.1f}%)")
            
    except Exception as e:
        print(f"Error reading batch file: {e}")

def scrape_upcoming_fixtures(league_name):
    """Scrapes upcoming, un-played fixtures."""
    # --- EUROPEAN LEAGUES (Use football-data to avoid 403 blocks and match team names) ---
    if league_name in EURO_LEAGUES:
        url = "https://www.football-data.co.uk/fixtures.csv"
        print(f"Downloading upcoming fixtures from {url}...")
        try:
            df = pd.read_csv(url)
            div_codes = EURO_LEAGUES[league_name]
            df = df[df['Div'].isin(div_codes)]
            
            if df.empty:
                return None
                
            fixtures = pd.DataFrame({
                'Home': df['HomeTeam'],
                'Away': df['AwayTeam']
            })
            return fixtures
        except Exception as e:
            print(f"Fixture Download Error: {e}")
            return None
            
    # --- CUSTOM LEAGUES (Fallback to fbref.com with enhanced stealth headers) ---
    elif league_name in FBREF_LEAGUE_IDS:
        comp_id = FBREF_LEAGUE_IDS[league_name]
        url = f"https://fbref.com/en/comps/{comp_id}/schedule/Scores-and-Fixtures"
        print(f"Scraping fixtures from {url}...")
        time.sleep(2)
    
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5"
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            tables = pd.read_html(response.text)
            df = tables[0]
            
            df = df[df['Score'].isna()]
            df = df[pd.to_numeric(df['Wk'], errors='coerce').notna()]
            
            if df.empty:
                return None
    
            fixtures = df[['Home', 'Away']].copy()
            return fixtures
            
        except Exception as e:
            print(f"Fixture Scraping Error: {e}")
            return None
    else:
        print(f"Error: Fixture scraping not supported for '{league_name}'.")
        return None

def run_fixture_predictions(league_name):
    """Scrapes and predicts all upcoming fixtures for a league."""
    fixtures_df = scrape_upcoming_fixtures(league_name)

    if fixtures_df is None or fixtures_df.empty:
        print("Could not find any upcoming fixtures for this league.")
        return

    print(f"\n=== PREDICTING {len(fixtures_df)} UPCOMING MATCHES FOR {league_name.upper()} ===")
    for index, row in fixtures_df.iterrows():
        home = str(row['Home']).strip()
        away = str(row['Away']).strip()
        
        probs = get_match_probabilities(league_name, home, away, verbose=False)
        if not probs:
            print(f"\n- {home} vs {away} Could not generate prediction (check team names/data).")
            continue
        
        markets = {
            "Home Win (1)": probs['home'], "Away Win (2)": probs['away'],
            "Home DNB (DNB1)": probs['dnb_home'], "Away DNB (DNB2)": probs['dnb_away'],
            "Double Chance (1X)": probs['home'] + probs['draw'], "Double Chance (X2)": probs['away'] + probs['draw'],
            "Over 1.5 Goals": probs['o15'], "Over 2.5 Goals": probs['o25'], 
            "Under 2.5 Goals": 1 - probs['o25'], "BTTS (Yes)": probs['btts'],
            "Home -1.5": probs['h_minus_1_5'], "Away -1.5": probs['a_minus_1_5']
        }
        safest_market = max(markets, key=markets.get)
        safest_prob = markets[safest_market] * 100
        
        print(f"\n- {home} vs {away}")
        print(f"  Safest Bet: {safest_market} ({safest_prob:.1f}%)")
        print(f"  (1: {probs['home']*100:.1f}% | 2: {probs['away']*100:.1f}% | DNB1: {probs['dnb_home']*100:.1f}% | O1.5: {probs['o15']*100:.1f}%)")

def import_custom_results(league_name, filepath):
    print(f"--- IMPORTING RESULTS TO {league_name.upper()} ---")
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        return

    new_rows = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                # Custom format: "Date","Status","Home","FTHG","FTAG","Away","Attd"
                # We only want to process rows where Status is "FT"
                if len(row) >= 6 and row[1].strip() == "FT":
                    home = row[2].strip()
                    fthg = row[3].strip()
                    ftag = row[4].strip()
                    away = row[5].strip()
                    
                    if fthg.isdigit() and ftag.isdigit():
                        new_rows.append({'HomeTeam': home, 'AwayTeam': away, 'FTHG': int(fthg), 'FTAG': int(ftag)})
                        
        if not new_rows:
            print("No valid 'FT' (Full Time) match results found in the file.")
            return
            
        new_df = pd.DataFrame(new_rows)
        target_file = os.path.join(DATA_DIR, f"{league_name}.csv")
        
        if os.path.exists(target_file):
            existing_df = pd.read_csv(target_file)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df = combined_df.fillna(0)
            combined_df.to_csv(target_file, index=False)
            print(f"Success! Added {len(new_df)} new match results into {target_file}.")
    except Exception as e:
        print(f"Error reading custom results file: {e}")

# ==========================================
#      MAIN MENU
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description="A football prediction and analysis tool.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="mode", help="Available modes", required=True)

    # --- Update command ---
    valid_leagues = list(EURO_LEAGUES.keys()) + list(CUSTOM_LEAGUES.keys())
    parser_update = subparsers.add_parser("update", help="Update league data from a CSV or by scraping.")
    parser_update.add_argument("league", help=f"The league to update.\nChoices: {', '.join(valid_leagues)}")

    # --- Predict command ---
    parser_predict = subparsers.add_parser("predict", help="Predict a single match outcome.")
    parser_predict.add_argument("league", help="The league of the match.")
    parser_predict.add_argument("home", help="The home team name.")
    parser_predict.add_argument("away", help="The away team name.")

    # --- Slip command ---
    subparsers.add_parser("slip", help="Interactively build a bet slip and check its safety.")

    # --- Batch command ---
    parser_batch = subparsers.add_parser("batch", help="Predict multiple matches from a CSV file.")
    parser_batch.add_argument("file", help="Path to the CSV file containing fixtures.")
    parser_batch.add_argument("--date", help="Optional: Filter matches by a specific date or week (requires a 'Date' column in CSV).", default=None)

    # --- Fixtures command ---
    parser_fixtures = subparsers.add_parser("fixtures", help="Scrape upcoming fixtures for a league and predict them.")
    parser_fixtures.add_argument("league", help=f"The league to get fixtures for.\nChoices: {', '.join(FBREF_LEAGUE_IDS.keys())}")

    # --- Import command ---
    parser_import = subparsers.add_parser("import", help="Import custom match results from a CSV file.")
    parser_import.add_argument("league", help="The league to update (e.g., epl).")
    parser_import.add_argument("file", help="Path to the CSV file containing the results.")

    args = parser.parse_args()

    if args.mode == "update":
        league = args.league.lower()
        if league in EURO_LEAGUES:
            update_euro_data(league)
        elif league in CUSTOM_LEAGUES:
            update_custom_data(league)
        else:
            print(f"Error: Unknown league '{league}'.")
    elif args.mode == "predict":
        run_single_prediction(args.league, args.home, args.away)
    elif args.mode == "slip":
        run_slip_checker()
    elif args.mode == "batch":
        run_batch_predictions(args.file, args.date)
    elif args.mode == "fixtures":
        run_fixture_predictions(args.league)
    elif args.mode == "import":
        import_custom_results(args.league, args.file)

if __name__ == "__main__":
    main()