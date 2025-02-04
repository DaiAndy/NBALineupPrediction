import torch
import pandas as pd
import lstm_model as mod

# maps the id back to the player name
id_to_player = {v: k for k, v in mod.player_to_id.items()}

# Load test dataset
df_test = pd.read_csv("matchups-2009.csv")



# Convert player names to numeric IDs
for col in ['home_0', 'home_1', 'home_2', 'home_3']:
    df_test[col] = df_test[col].map(mod.player_to_id)

X_new = torch.tensor(df_test[mod.input_features].values, dtype=torch.float32).to(mod.device)

# Load trained model
model = mod.LineupPredictor(len(mod.player_to_id), 32, 64, len(mod.input_features) - 4, 2).to(mod.device)
model.load_state_dict(torch.load("nba_lineup_model.pth"))
model.eval()

# Predict the 5th player
with torch.no_grad():
    player_inputs = X_new[:, :4].long()
    context_inputs = X_new[:, 4:]
    predictions = model(player_inputs, context_inputs)
    predicted_ids = torch.argmax(predictions, dim=1)

id_to_player = {v: k for k, v in mod.player_to_id.items()}
df_test["Predicted_5th_Player"] = [id_to_player[pred.item()] for pred in predicted_ids]

for col in ['home_0', 'home_1', 'home_2', 'home_3']:
    df_test[col] = df_test[col].map(id_to_player)

print(df_test[['home_0', 'home_1', 'home_2', 'home_3', 'Predicted_5th_Player']].head(10))

# saves the full output into a csv
df_test[['home_0', 'home_1', 'home_2', 'home_3', 'Predicted_5th_Player']].to_csv("predicted_lineups.csv", index=False)
print("redictions saved to predicted_lineups.csv")