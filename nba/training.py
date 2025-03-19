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

# loading the data into dataframe
df1 = pd.read_csv("matchups-2007.csv")
df2 = pd.read_csv("matchups-2008.csv")
df3 = pd.read_csv("matchups-2009.csv")
df4 = pd.read_csv("matchups-2010.csv")
df5 = pd.read_csv("matchups-2011.csv")
df6 = pd.read_csv("matchups-2012.csv")
df7 = pd.read_csv("matchups-2013.csv")
df8 = pd.read_csv("matchups-2014.csv")

# splitting the training data and the testing data
df_train = pd.concat([df1, df2], ignore_index=True)
df_test = df3
# ------------------------------------------------------------------------------------------------------------
# TRAINING

# function to sort players to their respected teams
teamArray = {}

# checks through all the home teams in training data
for _, row in df_train.iterrows():

    # home team row
    homeTeam = row["home_team"]

    # goes through each section of home_0 -> home_4
    for i in range(5):
        player = row[f'home_{i}']

        # adds player to the team in that array if it hasn't been added yet
        teamArray.setdefault(homeTeam, set()).add(player)

# checks through all the away teams in training data
for _, row in df_train.iterrows():

    # away team row
    awayTeam = row["away_team"]

    # goes through each player section of away_0 -> away_4
    for i in range(5):
        player = row[f'away_{i}']

        # adds player to the team in that array if it hasn't been added yet
        teamArray.setdefault(awayTeam, set()).add(player)

# checks through all the home teams in testing data
for _, row in df_test.iterrows():

    # home team row
    homeTeam = row["home_team"]

    # goes through each section of home_0 -> home_4
    for i in range(5):
        player = row[f'home_{i}']

        # adds player to the team in that array if it hasn't been added yet
        teamArray.setdefault(homeTeam, set()).add(player)

# checks through all the away teams in training data
for _, row in df_test.iterrows():

    # away team row
    awayTeam = row["away_team"]

    # goes through each player section of away_0 -> away_4
    for i in range(5):
        player = row[f'away_{i}']

        # adds player to the team in that array if it hasn't been added yet
        teamArray.setdefault(awayTeam, set()).add(player)

# prints output for show array working
print(teamArray)

# Select features (first 4 players)
features = ['season','home_team', 'away_team', 'starting_min', 'home_0', 'home_1', 'home_2', 'home_3', 'away_0', 'away_1', 'away_2', 'away_3', 'away_4']

df_train["label"] = df_train["home_4"]
df_test["label"] = df_test["home_4"]

if 'home_4' in features:
    features.remove('home_4')

df_train = df_train[features + ["label"]].copy()
df_test = df_test[features + ["label"]].copy()

# encoding teams and players
player_encode = LabelEncoder()
team_encode = LabelEncoder()

# fit encoders
player_cols = ['home_0','home_1','home_2','home_3','away_0','away_1','away_2','away_3','away_4','label']

train_cols_for_players = [col for col in player_cols if col in df_train.columns]
all_players_train = df_train[train_cols_for_players]

test_cols_for_players = [col for col in player_cols if col in df_test.columns]
all_players_test = df_test[test_cols_for_players]

all_players = pd.concat([all_players_train, all_players_test], ignore_index=True).values.flatten()

player_encode.fit(all_players.astype(str))

df_train['label'] = player_encode.transform(df_train['label'].astype(str))
df_test['label'] = player_encode.transform(df_test['label'].astype(str))

# teams and player names mapped to int ids
possible_player_cols = ['home_0','home_1','home_2','home_3','away_0','away_1','away_2','away_3','away_4']

for col in possible_player_cols:
    if col in df_train.columns:
        df_train[col] = player_encode.transform(df_train[col].astype(str))
    if col in df_test.columns:
        df_test[col] = player_encode.transform(df_test[col].astype(str))

# encoding teams
all_teams_train = df_train[["home_team", "away_team"]]
all_teams_test = df_test[["home_team", "away_team"]]
all_teams = pd.concat([all_teams_train, all_teams_test], ignore_index=True).values.flatten()

team_encode.fit(all_teams.astype(str))

df_train['home_team'] = team_encode.transform(df_train['home_team'].astype(str))
df_train['away_team'] = team_encode.transform(df_train['away_team'].astype(str))
df_test['home_team'] = team_encode.transform(df_test['home_team'].astype(str))
df_test['away_team'] = team_encode.transform(df_test['away_team'].astype(str))

# Normalize numeric features
df_train['starting_min'] = df_train['starting_min'] / 48.0
df_test['starting_min'] = df_test['starting_min'] / 48.0

season_min = df_train['season'].min()
season_max = df_train['season'].max()
df_train['season'] = (df_train['season'] - season_min) / (season_max - season_min)
df_test['season'] = (df_test['season'] - season_min) / (season_max - season_min)

X_train = df_train[features].values
y_train = df_train['label'].values

X_test = df_test[features].values
y_test = df_test['label'].values

class NBADataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = NBADataset(X_train, y_train)
test_dataset  = NBADataset(X_test,  y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

class nbaPredictor(nn.Module):
    def __init__(self, num_players, num_teams, embed_dim=32):
        super(nbaPredictor, self).__init__()
        # +1 for padding_idx, if you want to handle an extra "unknown" index
        self.player_embedding = nn.Embedding(num_players + 1, embed_dim, padding_idx=num_players)
        self.team_embedding = nn.Embedding(num_teams, embed_dim)

        # Features: 5 home players, 2 team embeddings, 2 numeric
        input_size = embed_dim * 5 + embed_dim * 2 + 2
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_players)
        )

    def forward(self, x, home_team):
        # The indexes in x must line up with the features array order:
        idx_season    = 0
        idx_home_team = 1
        idx_away_team = 2
        idx_start_min = 3
        home_idx = [4, 5, 6, 7, 8]  # the 5 columns for home players

        season = x[:, idx_season].float().unsqueeze(1)
        start_min = x[:, idx_start_min].float().unsqueeze(1)

        home_team_emb = self.team_embedding(x[:, idx_home_team])
        away_team_emb = self.team_embedding(x[:, idx_away_team])

        # Embed the 5 home player IDs
        home_players = torch.stack([x[:, i] for i in home_idx], dim=1)
        home_emb = self.player_embedding(home_players)  # (batch,5,embed_dim)
        home_emb = home_emb.view(home_emb.size(0), -1) # flatten => (batch, 5*embed_dim)

        combined = torch.cat([home_emb, home_team_emb, away_team_emb, season, start_min], dim=1)
        out = self.fc(combined)

        return out


# initializing the model
num_players = len(player_encode.classes_)
num_teams = len(team_encode.classes_)

model = nbaPredictor(num_players, num_teams).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 7
for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        home_team = inputs[:, 1]  # Index 1 is 'home_team' in X

        optimizer.zero_grad()
        outputs = model(inputs, home_team)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

print("Training complete")
torch.save(model.state_dict(), "nba_model.pth")
print("Model saved")