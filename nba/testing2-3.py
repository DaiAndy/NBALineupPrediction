import torch
import pandas as pd
import numpy as np
import pickle
import torch.nn as nn
from scipy.spatial.distance import cosine


# -----------------------------------------------------------
# 1) DEFINE MODEL
# -----------------------------------------------------------
class NBAPlayerPredictionModel(nn.Module):
    def __init__(self, input_size=61, hidden_size=512, output_size=50):
        super(NBAPlayerPredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


# -----------------------------------------------------------
# 2) LOAD THE TRAINED MODEL
# -----------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = NBAPlayerPredictionModel().to(device)
model.load_state_dict(torch.load("nba_lineup_model.pth"))


model.eval()
print("Model loaded successfully!")

# -----------------------------------------------------------
# 3) LOAD TEST DATA
# -----------------------------------------------------------
test_df = pd.read_csv("NBA_test.csv")

# -----------------------------------------------------------
# 4) LOAD PLAYER EMBEDDINGS
# -----------------------------------------------------------
with open("player_embeddings.pkl", "rb") as f:
    player_vectors = pickle.load(f)


# -----------------------------------------------------------
# 5) FEATURE FUNCTIONS
# -----------------------------------------------------------
def compute_lineup_embedding(players, embedding_dict):
    valid_players = [p for p in players if p in embedding_dict]
    return np.mean([embedding_dict[p] for p in valid_players], axis=0) if valid_players else np.zeros(50)


def categorize_starting_minute(minute):
    if minute < 12:
        return [1, 0, 0]
    elif minute < 24:
        return [0, 1, 0]
    else:
        return [0, 0, 1]


def find_closest_player(predicted_emb, candidates, embedding_dict):
    best_player = None
    best_similarity = float("-inf")
    for candidate in candidates:
        if candidate in embedding_dict:
            sim = 1 - cosine(predicted_emb, embedding_dict[candidate])
            if sim > best_similarity:
                best_similarity = sim
                best_player = candidate
    return best_player


# -----------------------------------------------------------
# 6) PREPARE TEST SAMPLES
# -----------------------------------------------------------
test_lineups = []
test_home_teams = []
missing_positions = []
row_indices = []
start_times = []

for idx, row in test_df.iterrows():
    home_team = row["home_team"]
    known_players = []
    missing_index = None
    for i in range(5):
        player = row[f"home_{i}"]
        if player == "?":
            missing_index = i
        else:
            known_players.append(player)
    if missing_index is not None:
        test_lineups.append(known_players)
        test_home_teams.append(home_team)
        missing_positions.append(missing_index)
        row_indices.append(idx)
        start_times.append(row.get("starting_min", 0))

print(f"Prepared {len(test_lineups)} test lineups for prediction.")

# -----------------------------------------------------------
# 7) RUN PREDICTIONS
# -----------------------------------------------------------
predictions = []

for lineup, home_team, missing_idx, row_idx, start_min in zip(test_lineups, test_home_teams, missing_positions,
                                                              row_indices, start_times):
    lineup_emb = compute_lineup_embedding(lineup, player_vectors)
    start_time_category = np.array(categorize_starting_minute(start_min))
    lineup_tensor = torch.tensor(np.hstack((lineup_emb, home_away_bias, start_time_category)),
                                 dtype=torch.float32).unsqueeze(0).to(device)
    print(f"Shape of lineup_tensor: {lineup_tensor.shape}")
    print(f"Shape of lineup_tensor before prediction: {lineup_tensor.shape}")
    with torch.no_grad():
        predicted_emb = model(lineup_tensor).cpu().numpy().flatten()

    predicted_player = find_closest_player(predicted_emb, player_vectors.keys(), player_vectors)

    predictions.append({
        "row_idx": row_idx,
        "Fifth_Player": predicted_player
    })

# -----------------------------------------------------------
# 8) SAVE PREDICTIONS
# -----------------------------------------------------------
predictions_df = pd.DataFrame(predictions)
predictions_df.to_csv("NBA_predictions.csv", index=False)
print("Predictions saved to 'NBA_predictions.csv'!")