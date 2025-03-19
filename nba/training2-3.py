import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import Word2Vec
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# 1) LOAD AND COMBINE DATA
# ---------------------------------------------------------------------------
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

# Optional: Filter or shuffle, then do your train/test split (80/20, etc.)
df_train = df_all.sample(frac=0.8, random_state=42)
# Or do a time-based split: older seasons for training, newer for test, etc.

# ---------------------------------------------------------------------------
# 2) BUILD WORD2VEC MODEL FOR PLAYER NAMES
#    (Lineup-based sentences for training Word2Vec)
# ---------------------------------------------------------------------------
all_lineups = []
for _, row in df_train.iterrows():
    # Each row has home_0..home_4, away_0..away_4
    # We can treat the "sentence" as the 10 players in a match
    home_players = [row[f"home_{i}"] for i in range(5)]
    away_players = [row[f"away_{i}"] for i in range(5)]
    all_lineups.append(home_players)
    all_lineups.append(away_players)

# Train a new Word2Vec on all_lineups
player_embedding_model = Word2Vec(
    sentences=all_lineups,
    vector_size=50,  # keep or adjust dimension
    window=3,
    min_count=1,
    workers=4
)

player_vectors = {
    player: player_embedding_model.wv[player]
    for player in player_embedding_model.wv.index_to_key
}

# ---------------------------------------------------------------------------
# 3) CREATE HOME/AWAY APPEARANCE COUNTS (FOR HOME/AWAY BIAS FEATURES)
# ---------------------------------------------------------------------------
home_appearances = {}
away_appearances = {}
for _, row in df_train.iterrows():
    for i in range(5):
        hp = row[f'home_{i}']
        ap = row[f'away_{i}']
        home_appearances[hp] = home_appearances.get(hp, 0) + 1
        away_appearances[ap] = away_appearances.get(ap, 0) + 1

def compute_home_away_bias(player):
    total = home_appearances.get(player, 0) + away_appearances.get(player, 0)
    if total == 0:
        return np.array([0.5, 0.5])
    return np.array([
        home_appearances.get(player, 0) / total,
        away_appearances.get(player, 0) / total
    ])

# ---------------------------------------------------------------------------
# 4) FEATURE-ENGINEERING FUNCTIONS
#    - We'll do the same as in testing2.py: lineup emb + home/away bias + start time
# ---------------------------------------------------------------------------
def compute_lineup_embedding(players, emb_dict):
    """Average the embeddings of known players."""
    valid = [p for p in players if p in emb_dict]
    if not valid:
        return np.zeros(50)
    return np.mean([emb_dict[p] for p in valid], axis=0)

def categorize_starting_minute(minute):
    """One-hot for Early (<12), Mid (<24), Late (>=24)."""
    if minute < 12:
        return np.array([1, 0, 0])
    elif minute < 24:
        return np.array([0, 1, 0])
    else:
        return np.array([0, 0, 1])

# ---------------------------------------------------------------------------
# 5) BUILD TRAINING SAMPLES
#    For each row, pretend one 'home' spot is missing, gather features => predicted embedding
# ---------------------------------------------------------------------------
X_train_list = []
y_train_list = []

for _, row in df_train.iterrows():
    # We'll only do this if we have a notion of "starting_min" or something similar:
    # If your dataset doesn't have it, you can default it to 0 or skip that part.
    start_min = row.get("starting_min", 0)  # or row["starting_min"] if guaranteed

    for i in range(5):
        missing_player = row[f"home_{i}"]
        # Known players are the other 4 spots
        known_players = [row[f"home_{j}"] for j in range(5) if j != i]

        # Check if we have embeddings for missing_player
        if missing_player not in player_vectors:
            continue  # skip if we can't get an embedding

        # 1) lineup_emb: average embedding (50D)
        lineup_emb = compute_lineup_embedding(known_players, player_vectors)

        # 2) home/away bias (2D for each known player => up to 8 or 10 dims)
        #   If we have exactly 4 known players, that's 4*2 = 8 dims.
        #   If you want to fix dimension as 10, you can do a small loop+pad
        habias_list = []
        for kp in known_players:
            habias_list.append(compute_home_away_bias(kp))
        # If fewer than 5 players known, pad:
        while len(habias_list) < 5:
            habias_list.append([0.5, 0.5])
        # But we only had 4 known (since 1 is missing), so let's do len=4:
        # Or we can always assume 4 known if exactly 1 is missing:
        habias_np = np.array(habias_list[:4]).flatten()
        # That yields 8D. If you REALLY want 10D, you'd handle it differently.

        # 3) Start time category (3D)
        start_time_cat = categorize_starting_minute(start_min)

        # Combine them. We'll do (50 + 8 + 3) = 61 dims here
        combined = np.hstack([lineup_emb, habias_np, start_time_cat])

        # We'll store the "true" embedding for the missing player as the label
        true_emb = player_vectors[missing_player]

        X_train_list.append(combined)
        y_train_list.append(true_emb)

# Convert to tensors
X_train_tensor = torch.tensor(X_train_list, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_list, dtype=torch.float32).to(device)

print("Training samples:", len(X_train_tensor))

# ---------------------------------------------------------------------------
# 6) DEFINE MODEL: We'll unify with what you did in testing (similar style)
#    But note we have ~61 input dims in this example, not 63. Adjust as needed
# ---------------------------------------------------------------------------
class NBAPlayerPredictionModel(nn.Module):
    def __init__(self, input_size=61, hidden_size=512, output_size=50):
        super().__init__()
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

model = NBAPlayerPredictionModel().to(device)

# ---------------------------------------------------------------------------
# 7) TRAINING LOOP WITH COSINE LOSS
# ---------------------------------------------------------------------------
def cosine_loss(pred, target):
    # 1 - cos similarity
    # pred, target shape: [batch_size, 50]
    cos_sim = F.cosine_similarity(pred, target, dim=1)
    return 1 - cos_sim.mean()  # want to maximize average cos, so minimize (1 - cos)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
batch_size = 64
epochs = 30  # train longer than 10, or tune as needed

dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    total_loss = 0
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = cosine_loss(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

# Finally, save the model + the new embeddings
torch.save(model.state_dict(), "nba_lineup_model.pth")
with open("player_embeddings.pkl", "wb") as f:
    pickle.dump(player_vectors, f)

print("Training complete. Model and embeddings saved.")
