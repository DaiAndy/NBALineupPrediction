import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

###############################################################################
# 2. Load Your CSV Data (Example: 5 columns for home players)
###############################################################################
# Let's assume you have at least these columns: home_0, home_1, home_2, home_3, home_4
df = pd.read_csv("matchups-2007.csv")  # or combine multiple years if you want

# Drop rows if any player is missing or NaN
df.dropna(subset=["home_0", "home_1", "home_2", "home_3", "home_4"], inplace=True)

###############################################################################
# 3. Create (known_players, missing_player) pairs
#    We'll produce 5 samples per row => each row has 5 players
###############################################################################
all_samples = []
for _, row in df.iterrows():
    players = [row[f"home_{i}"] for i in range(5)]
    # For each player in these 5, treat that player as "missing"
    # and the other 4 as "known"
    for i in range(5):
        known = [p for j, p in enumerate(players) if j != i]
        missing = players[i]
        all_samples.append((known, missing))

print("Total training samples:", len(all_samples))

###############################################################################
# 4. Label-Encode Player IDs into integer indices
###############################################################################
all_players = []
for known, missing in all_samples:
    all_players.extend(known)
    all_players.append(missing)

player_encoder = LabelEncoder()
player_encoder.fit(all_players)


def encode_player(p):
    return player_encoder.transform([p])[0]


# Example usage: encode_player("LeBron James") => 1234 (some integer)

###############################################################################
# 5. Build a PyTorch Dataset
###############################################################################
class MissingPlayerDataset(Dataset):
    def __init__(self, samples, encoder):
        """
        samples: list of (known_list_of_4, missing_player)
        encoder: LabelEncoder for players
        """
        self.X = []
        self.y = []
        for known, missing in samples:
            # Encode each of the 4 known players
            known_ids = [encoder.transform([p])[0] for p in known]
            # Encode missing player
            missing_id = encoder.transform([missing])[0]
            self.X.append(known_ids)
            self.y.append(missing_id)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.long)


dataset = MissingPlayerDataset(all_samples, player_encoder)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

num_players = len(player_encoder.classes_)  # total distinct players


###############################################################################
# 6. Define a Model with nn.Embedding for Players
###############################################################################
class MissingPlayerModel(nn.Module):
    def __init__(self, num_players, embed_dim=32):
        super(MissingPlayerModel, self).__init__()
        self.embed = nn.Embedding(num_players, embed_dim)
        # We have 4 known players => we’ll average their embeddings => embed_dim
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_players)  # output is logits over all players
        )

    def forward(self, x):
        """
        x shape: (batch_size, 4)  # 4 known players
        """
        # shape: (batch_size, 4, embed_dim)
        embeds = self.embed(x)
        # average across the 4 players => (batch_size, embed_dim)
        avg_emb = embeds.mean(dim=1)
        # pass through fc => (batch_size, num_players)
        out = self.fc(avg_emb)
        return out


model = MissingPlayerModel(num_players, embed_dim=32).to(device)

###############################################################################
# 7. Training Loop (CrossEntropy for classification)
###############################################################################
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)  # shape: [batch_size, 4]
        batch_y = batch_y.to(device)  # shape: [batch_size]

        optimizer.zero_grad()
        outputs = model(batch_x)  # shape: [batch_size, num_players]
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

# Done training!
torch.save(model.state_dict(), "missing_player_model.pth")
print("Model saved to missing_player_model.pth")
