import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

# checks if cuda is available to be used, if not defaults to cpu usage
print("Cuda:", torch.__version__, torch.cuda.current_device(), torch.version.cuda, torch.cuda.get_device_name(0))

# use cuda or use cpu and lets user know which one is being used
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"currently using: Cuda")
else:
    device = torch.device("CPU")
    print(f"currently using: CPU")

# Suppose we have a new CSV "NBA_test.csv" to fill
df_test = pd.read_csv("NBA_test.csv")

features = ['season','home_team', 'away_team', 'starting_min', 'home_0', 'home_1', 'home_2', 'home_3', 'away_0', 'away_1', 'away_2', 'away_3', 'away_4']

df_test = df_test[features + ["label"]].copy()

# encoding teams and players
player_encode = LabelEncoder()
team_encode = LabelEncoder()

# fit encoders
player_cols = ['home_0','home_1','home_2','home_3','away_0','away_1','away_2','away_3','away_4','label']

test_cols_for_players = [col for col in player_cols if col in df_test.columns]
all_players_test = df_test[test_cols_for_players]

all_players = pd.concat([all_players_test], ignore_index=True).values.flatten()

player_encode.fit(all_players.astype(str))

df_test['label'] = player_encode.transform(df_test['label'].astype(str))

# teams and player names mapped to int ids
possible_player_cols = ['home_0','home_1','home_2','home_3','away_0','away_1','away_2','away_3','away_4']

for col in possible_player_cols:
    if col in df_test.columns:
        df_test[col] = player_encode.transform(df_test[col].astype(str))

# encoding teams
all_teams_test = df_test[["home_team", "away_team"]]
all_teams = pd.concat([all_teams_test], ignore_index=True).values.flatten()

team_encode.fit(all_teams.astype(str))

df_test['home_team'] = team_encode.transform(df_test['home_team'].astype(str))
df_test['away_team'] = team_encode.transform(df_test['away_team'].astype(str))

# 1) Same transformations as training
df_test["home_team"] = team_encode.transform(df_test["home_team"].astype(str))
df_test["away_team"] = team_encode.transform(df_test["away_team"].astype(str))

player_cols = ["home_0","home_1","home_2","home_3","away_0","away_1","away_2","away_3","away_4"]
for c in player_cols:
    if c in df_test.columns:
        df_test[c] = player_encode.transform(df_test[c].astype(str))

season_min = df_test['season'].min()
season_max = df_test['season'].max()
df_test['season'] = (df_test['season'] - season_min) / (season_max - season_min)
df_test['season'] = (df_test['season'] - season_min) / (season_max - season_min)

# 2) Convert to tensor
X_infer = df_test[features].values
X_infer = torch.tensor(X_infer, dtype=torch.long).to(device)

# 3) Run model
model.eval()
batch_size = 64
predictions = []
for i in range(0, len(X_infer), batch_size):
    batch_X = X_infer[i : i + batch_size]
    home_team = batch_X[:, 1]  # if index=1 is indeed 'home_team'
    with torch.no_grad():
        out = model(batch_X, home_team)  # shape: [batch_size, num_players]
    pred_ids = torch.argmax(out, dim=1).tolist()
    predictions.extend(pred_ids)

# 4) Convert from numeric IDs back to string player names
pred_player_strings = player_encode.inverse_transform(predictions)

# 5) Add column to df_infer
df_test["predicted_home_4"] = pred_player_strings

# 6) Save new CSV
df_test.to_csv("NBA_test_filled.csv", index=False)
print("Predictions saved to NBA_test_filled.csv")
