import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from gensim.models import Word2Vec
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
df1 = pd.read_csv("matchups-2007.csv")
df2 = pd.read_csv("matchups-2008.csv")
df3 = pd.read_csv("matchups-2009.csv")
df4 = pd.read_csv("matchups-2010.csv")
df5 = pd.read_csv("matchups-2011.csv")
df6 = pd.read_csv("matchups-2012.csv")
df7 = pd.read_csv("matchups-2013.csv")
df8 = pd.read_csv("matchups-2014.csv")
df9 = pd.read_csv("matchups-2015.csv")

df_all = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9], ignore_index=True)

# Train-test split (80% train, 20% test)
df_train = df_all.sample(frac=0.8, random_state=42)
df_test = df_all.drop(df_train.index)

# Ensure no overlapping players between train and test
train_players = set(df_train['home_0']).union(df_train['home_1'], df_train['home_2'], df_train['home_3'],
                                              df_train['home_4'])
df_test = df_test[df_test.apply(lambda row: not any(row[f'home_{i}'] in train_players for i in range(5)), axis=1)]

# Track players by team and season
team_season_players = {}

for _, row in df_train.iterrows():
    season = row['season']
    home_team = row['home_team']
    away_team = row['away_team']

    for i in range(5):
        player_home = row[f'home_{i}']
        player_away = row[f'away_{i}']

        team_season_players.setdefault((home_team, season), set()).add(player_home)
        team_season_players.setdefault((away_team, season), set()).add(player_away)


# Function to get valid players
def get_valid_players(home_team, season):
    """Retrieve players from the home team who played in the current or past seasons."""
    valid_players = set()
    for past_season in range(2007, season + 1):  # Include all seasons up to the current one
        valid_players.update(team_season_players.get((home_team, past_season), set()))
    return valid_players


# Collect training data
currentLineup = []
missing_players = []
for _, row in df_train.iterrows():
    for i in range(5):  # Iterate through each position
        known_players = [row[f'home_{j}'] for j in range(5) if j != i]
        missing_player = row[f'home_{i}']

        if missing_player and all(known_players):
            currentLineup.append(known_players)
            missing_players.append(missing_player)

# Train Word2Vec model
player_embedding_model = Word2Vec(sentences=currentLineup, vector_size=50, window=3, min_count=1, workers=4)
player_vectors = {player: player_embedding_model.wv[player] for player in player_embedding_model.wv.index_to_key}

# Ensure correct alignment of X_train and y_train
valid_data = [(players, player) for players, player in zip(currentLineup, missing_players) if player in player_vectors]
X_train, y_train = zip(*valid_data)
X_train = np.array([np.mean([player_vectors[p] for p in players], axis=0) for players in X_train])
y_train = np.array([player_vectors[player] for player in y_train])

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)


# Define Model
class NBAPlayerPredictionModel(nn.Module):
    def __init__(self, input_size=50, hidden_size=128, output_size=50):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# Training
model = NBAPlayerPredictionModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

for epoch in range(10):
    total_loss = 0
    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch + 1}/10], Loss: {total_loss / len(dataloader):.4f}")

torch.save(model.state_dict(), "nba_lineup_model.pth")


# Predicting missing player with filtered candidates
def predict_best_match(model, X, home_team, season):
    """Predict the missing player from valid players (home team + past seasons)."""
    pred_embedding = model(X).cpu().detach().numpy()
    valid_players = get_valid_players(home_team, season)

    # Convert valid player names to embeddings
    valid_player_vectors = {p: player_vectors[p] for p in valid_players if p in player_vectors}

    if not valid_player_vectors:
        return "Unknown Player"  # No valid players found

    # Find the closest match
    best_match = min(valid_player_vectors.keys(),
                     key=lambda p: np.linalg.norm(valid_player_vectors[p] - pred_embedding))
    return best_match


# Test and Evaluate
print("\nEvaluating Model on Test Set...\n")

for _, row in df_test.iterrows():
    home_team = row['home_team']
    season = row['season']
    known_players = [row[f'home_{i}'] for i in range(4) if row[f'home_{i}'] in player_vectors]  # Use first 4 players

    if len(known_players) == 0:
        print(f"Skipping row {row.name} due to missing player embeddings")
        continue

    # Compute test lineup embedding
    X_test = np.mean([player_vectors[p] for p in known_players], axis=0)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(0).to(device)

    predicted_player = predict_best_match(model, X_test_tensor, home_team, season)
    actual_player = row['home_4']

    print(f"Predicted: {predicted_player}, Actual: {actual_player}")

print("Evaluation Complete!")
