import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# loading the data
train1 = "matchups-2007.csv"
train2 = "matchups-2008.csv"

# test data
test = "matchups-2009.csv"

df = pd.read_csv(train1)
df2 = pd.read_csv(train2)
df_test = pd.read_csv(test)

# Select features (first 4 players + game context)
input_features = ['home_0', 'home_1', 'home_2', 'home_3', 'starting_min', 'end_min',
                  'fga_home', 'fta_home', 'fgm_home', 'ast_home', 'reb_home', 'to_home', 'pts_home']
target_feature = 'home_4'  # Predicting the 5th player

df_train = pd.concat([df, df2], ignore_index=True)

# Define player columns
player_columns = ['home_0', 'home_1', 'home_2', 'home_3', 'home_4',
                  'away_0', 'away_1', 'away_2', 'away_3', 'away_4']

# Create a mapping of unique players to numeric IDs
unique_players = pd.unique(pd.concat([df_train[player_columns], df_test[player_columns]], axis=0).values.ravel())
player_to_id = {player: idx for idx, player in enumerate(unique_players)}

# Convert player names to numeric IDs
for col in player_columns:
    df_train[col] = df_train[col].map(player_to_id)
    df_test[col] = df_test[col].map(player_to_id)


# Split dataset
#train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(df_train[input_features].values, dtype=torch.float32)
y_train = torch.tensor(df_train[target_feature].values, dtype=torch.long)
X_test = torch.tensor(df_test[input_features].values, dtype=torch.float32)
y_test = torch.tensor(df_test[target_feature].values, dtype=torch.long)

# Convert to tensors// tests against itself
#X_train = torch.tensor(train_data[input_features].values, dtype=torch.float32)
#y_train = torch.tensor(train_data[target_feature].values, dtype=torch.long)
#X_test = torch.tensor(test_data[input_features].values, dtype=torch.float32)
#y_test = torch.tensor(test_data[target_feature].values, dtype=torch.long)

#-------------------------------------------------------------------------------

print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move data to GPU
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

class NBALineupDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Define batch size
batch_size = 64
train_loader = DataLoader(NBALineupDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(NBALineupDataset(X_test, y_test), batch_size=batch_size, shuffle=False)


# Define model
class LineupPredictor(nn.Module):
    def __init__(self, num_players, embed_dim, hidden_dim, context_dim, num_layers):
        super(LineupPredictor, self).__init__()
        self.embedding = nn.Embedding(num_players, embed_dim)  # Player embeddings
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_players)
        )

    def forward(self, player_ids, context_features):
        embedded = self.embedding(player_ids)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = lstm_out[:, -1, :]
        combined = torch.cat((lstm_out, context_features), dim=1)
        return self.fc(combined)


if __name__ == "__main__":

    # Initialize model
    num_players = len(player_to_id)
    embed_dim = 32
    hidden_dim = 64
    context_dim = len(input_features) - 4  # Exclude player columns
    num_layers = 2

    model = LineupPredictor(num_players, embed_dim, hidden_dim, context_dim, num_layers).to(device)

    # Define loss function & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            player_inputs = inputs[:, :4].long()
            context_inputs = inputs[:, 4:]

            optimizer.zero_grad()
            outputs = model(player_inputs, context_inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

    print("Training complete")
    torch.save(model.state_dict(), "nba_lineup_model.pth")
    print("Model saved")
