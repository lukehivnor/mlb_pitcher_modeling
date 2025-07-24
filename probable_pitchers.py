import requests
import pandas as pd
from datetime import datetime, timedelta

def get_upcoming_starters(days=2):
    """
    Fetches the next `days` days of MLB games and their probable pitchers
    from the MLB Stats API.
    """
    records = []
    base_url = "https://statsapi.mlb.com/api/v1/schedule"
    for delta in range(days):
        date = (datetime.now() + timedelta(days=delta)).strftime("%Y-%m-%d")
        params = {
            "sportId": 1,
            "date": date,
            "hydrate": "probablePitcher"
        }
        resp = requests.get(base_url, params=params)
        resp.raise_for_status()
        data = resp.json()

        for date_info in data.get("dates", []):
            for game in date_info.get("games", []):
                away = game["teams"]["away"]["team"]["name"]
                home = game["teams"]["home"]["team"]["name"]
                away_prob = game["teams"]["away"].get("probablePitcher") or {}
                home_prob = game["teams"]["home"].get("probablePitcher") or {}

                records.append({
                    "date":      date,
                    "away_team": away,
                    "home_team": home,
                    "away_pitcher": away_prob.get("fullName"),
                    "away_id":      away_prob.get("id"),
                    "home_pitcher": home_prob.get("fullName"),
                    "home_id":      home_prob.get("id"),
                    "game_time":    game.get("gameDate")
                })

    return pd.DataFrame(records)

if __name__ == "__main__":
    df = get_upcoming_starters(days=2)
    print(df)
