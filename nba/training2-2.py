# Group #8

# imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from gensim.models import Word2Vec
import pickle

# checks if cuda is available to be used, if not defaults to cpu usage
print("Cuda:", torch.__version__, torch.cuda.current_device(), torch.version.cuda, torch.cuda.get_device_name(0))

# use cuda or use cpu and lets user know which one is being used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# importing all csv files to be used for training and testing
df2007 = pd.read_csv("matchups-2007.csv")
df2008 = pd.read_csv("matchups-2008.csv")
df2009 = pd.read_csv("matchups-2009.csv")
df2010 = pd.read_csv("matchups-2010.csv")
df2011 = pd.read_csv("matchups-2011.csv")
df2012 = pd.read_csv("matchups-2012.csv")
df2013 = pd.read_csv("matchups-2013.csv")
df2014 = pd.read_csv("matchups-2014.csv")
df2015 = pd.read_csv("matchups-2015.csv")
df_list = [pd.read_csv(f"matchups-{year}.csv") for year in range(2007, 2016)]
df_all = pd.concat(df_list, ignore_index=True)
df_train = df_all[df_all["season"] <= 2013]
df_test = df_all[df_all["season"] >= 2014]

# holds all available player in each season for each home/away team
season2007Array = {}
season2008Array = {}
season2009Array = {}
season2010Array = {}
season2011Array = {}
season2012Array = {}
season2013Array = {}
season2014Array = {}
season2015Array = {}

# --------------------- preprocessing data ------------------------------
# checks through all the home teams
for _, row in df2007.iterrows():
    # home team row
    homeTeam = row["home_team"]
    awayTeam = row["away_team"]
    # goes through each section of home_0 -> home_4
    for i in range(5):
        player = row[f'home_{i}']
        player2 = row[f'away_{i}']
        # adds player to the team in that array if it hasn't been added yet
        season2007Array.setdefault(homeTeam, set()).add(player)
        season2007Array.setdefault(homeTeam, set()).add(player2)

# checks through all the home teams
for _, row in df2008.iterrows():
    # home team row
    homeTeam = row["home_team"]
    awayTeam = row["away_team"]
    # goes through each section of home_0 -> home_4
    for i in range(5):
        player = row[f'home_{i}']
        player2 = row[f'away_{i}']
        # adds player to the team in that array if it hasn't been added yet
        season2008Array.setdefault(homeTeam, set()).add(player)
        season2008Array.setdefault(homeTeam, set()).add(player2)

# checks through all the home teams
for _, row in df2009.iterrows():
    # home team row
    homeTeam = row["home_team"]
    awayTeam = row["away_team"]
    # goes through each section of home_0 -> home_4
    for i in range(5):
        player = row[f'home_{i}']
        player2 = row[f'away_{i}']
        # adds player to the team in that array if it hasn't been added yet
        season2009Array.setdefault(homeTeam, set()).add(player)
        season2009Array.setdefault(homeTeam, set()).add(player2)

for _, row in df2010.iterrows():
    # home team row
    homeTeam = row["home_team"]
    awayTeam = row["away_team"]
    # goes through each section of home_0 -> home_4
    for i in range(5):
        player = row[f'home_{i}']
        player2 = row[f'away_{i}']
        # adds player to the team in that array if it hasn't been added yet
        season2010Array.setdefault(homeTeam, set()).add(player)
        season2010Array.setdefault(homeTeam, set()).add(player2)

for _, row in df2011.iterrows():
    # home team row
    homeTeam = row["home_team"]
    awayTeam = row["away_team"]
    # goes through each section of home_0 -> home_4
    for i in range(5):
        player = row[f'home_{i}']
        player2 = row[f'away_{i}']
        # adds player to the team in that array if it hasn't been added yet
        season2011Array.setdefault(homeTeam, set()).add(player)
        season2011Array.setdefault(homeTeam, set()).add(player2)

for _, row in df2012.iterrows():
    # home team row
    homeTeam = row["home_team"]
    awayTeam = row["away_team"]
    # goes through each section of home_0 -> home_4
    for i in range(5):
        player = row[f'home_{i}']
        player2 = row[f'away_{i}']
        # adds player to the team in that array if it hasn't been added yet
        season2012Array.setdefault(homeTeam, set()).add(player)
        season2012Array.setdefault(homeTeam, set()).add(player2)

for _, row in df2013.iterrows():
    # home team row
    homeTeam = row["home_team"]
    awayTeam = row["away_team"]
    # goes through each section of home_0 -> home_4
    for i in range(5):
        player = row[f'home_{i}']
        player2 = row[f'away_{i}']
        # adds player to the team in that array if it hasn't been added yet
        season2013Array.setdefault(homeTeam, set()).add(player)
        season2013Array.setdefault(homeTeam, set()).add(player2)

for _, row in df2014.iterrows():
    # home team row
    homeTeam = row["home_team"]
    awayTeam = row["away_team"]
    # goes through each section of home_0 -> home_4
    for i in range(5):
        player = row[f'home_{i}']
        player2 = row[f'away_{i}']
        # adds player to the team in that array if it hasn't been added yet
        season2014Array.setdefault(homeTeam, set()).add(player)
        season2014Array.setdefault(homeTeam, set()).add(player2)

for _, row in df2015.iterrows():

    # home team row
    homeTeam = row["home_team"]
    awayTeam = row["away_team"]

    # goes through each section of home_0 -> home_4
    for i in range(5):
        player = row[f'home_{i}']
        player2 = row[f'away_{i}']
        # adds player to the team in that array if it hasn't been added yet
        season2015Array.setdefault(homeTeam, set()).add(player)
        season2015Array.setdefault(homeTeam, set()).add(player2)

# setting up two arrays, one to hold the home games and the other to hold the away games
homeGames = {}
awayGames = {}

for _, row in df_train.iterrows():
    for i in range(5):
        # goes through the csv file, finds the label of home_0 -> home_4 / away_0 -> away_4 and stores it into the respected arrays
        homeGames[row[f'home_{i}']] = homeGames.get(row[f'home_{i}'], 0) + 1
        awayGames[row[f'away_{i}']] = awayGames.get(row[f'away_{i}'], 0) + 1

# function to limit the players the model has access to predict with, regulating it to the three previous seasons
def possibleTeamPlayers(season, team):

    # if 2007, use itself, 2008 use 2007, and 2009, use 2008 and 2007, else every other season takes 3 seasons before it
    if season < 2007:
        return season2007Array.get(team, set())
    elif season == 2008:
        return season2007Array.get(team, set())
    elif season == 2009:
        return season2008Array.get(team, set()).union(season2007Array.get(team, set()))

    # fetches the dictionaries of each of the season being requested
    oneSeasonAgo = globals().get(f"season{season - 1}Array", {})
    twoSeasonAgo = globals().get(f"season{season - 2}Array", {})
    threeSeasonAgo = globals().get(f"season{season - 3}Array", {})

    oneSeasonsAgoPlayers = oneSeasonAgo.get(team, set())
    twoSeasonsAgoPlayers = twoSeasonAgo.get(team, set())
    threeSeasonsAgoPlayers = threeSeasonAgo.get(team, set())

    # returns a dictionary of the three seasons merged in with no duplicates
    return oneSeasonsAgoPlayers.union(twoSeasonsAgoPlayers).union(threeSeasonsAgoPlayers)


# --------------------- pre-processing ------------------------------

# array to hold the all team members in their respected team
teamArray = {}

# function to sort players to their respected teams and stores it into the teamArray
def process_team_players(df):
    for _, row in df.iterrows():
        home_team, away_team = row["home_team"], row["away_team"]
        for i in range(5):
            teamArray.setdefault(home_team, set()).add(row[f'home_{i}'])
            teamArray.setdefault(away_team, set()).add(row[f'away_{i}'])

process_team_players(df_train)
process_team_players(df_test)

# prints out the array of teams and the players on those teams just to check that its working properly
print(teamArray)

# setting up array to hold the current lineup and the missing players
currentLineup = []
missing_players = []
startingMinutes = []

# builds the training sample that will be used for training the model
for _, row in df_train.iterrows():
    season = row["season"]
    home_team = row["home_team"]

    # grabs a list of all the possible players for that home team/season
    possiblePlayers = possibleTeamPlayers(season, home_team)

    # each iteration pretends one player is missing and tries to predict it
    for i in range(5):

        # known players are checked if they car possible players
        visiblePlayers = [row[f'home_{j}'] for j in range(5) if j != i and row[f'home_{j}'] in possiblePlayers]
        # assigns the unknown to that home_0 -> 4 position
        unknown = row[f'home_{i}']

        # checks if unknown is eligible for the home team
        if unknown in possiblePlayers and all(p in possiblePlayers for p in visiblePlayers):
            currentLineup.append(visiblePlayers)
            missing_players.append(unknown)
            startingMinutes.append(row["starting_min"])

print(f"Total training samples: {len(currentLineup)}")

# using the Word2Vec model for embedding the player
playerEmbedding = Word2Vec(sentences=currentLineup, vector_size=50, window=5, min_count=1, workers=4)
playerVectors = {p: playerEmbedding.wv[p] for p in playerEmbedding.wv.index_to_key}
print(f"Total Players: {len(playerVectors)}")

# this will save the player embedding which will be needed for the testing file
with open("playerEmbeddings.pkl", "wb") as f:
    pickle.dump(playerVectors, f)

# function is used to calcualte the average distance for the word2vec embedding and returns it
def avgLineupEmbedding(players, model):
    possiblePlayer = [p for p in players if p in model.wv]

    if not possiblePlayer:
        return np.zeros(model.vector_size)

    # returns the average of all the possible players' embeddings
    return np.mean([model.wv[p] for p in possiblePlayer], axis=0)

# function is used to calcualte a 2D vector for the requested player
def homeAwayBias(player):

    # retrieves the times the player has played at home and away
    totalGames = homeGames.get(player, 0) + awayGames.get(player, 0)

    # returns the homegames / total games and awaygames/total games, which will show a bias that can be used
    return [homeGames.get(player, 0) / totalGames, awayGames.get(player, 0) / totalGames]

# takes the start minute and categorizes it
def timeCategorizing(time):

    # < 12 is early game, < 24 is mid-game, anything above is late game
    if time < 12:
        return [1, 0, 0]
    elif time < 24:
        return [0, 1, 0]
    else:
        return [0, 0, 1]
# setting up arrays to hold the data of each for the current lineup
homeAwayBiasArr = []
timeCategoryArr = []

# loops through each player and their starting minute
for players, startTime in zip(currentLineup, startingMinutes):

    # calls home away bias function and assigns it to bias
    bias = [homeAwayBias(p) for p in players]

    # flattens the array to be used
    bias = np.array(bias).flatten()

    # ensures that all players have a bias, if not, a default value is assigned to ensure that the code continues to run
    while len(bias) < 10:
        bias = np.append(bias, [0.5, 0.5])

    homeAwayBiasArr.append(bias)

    # calls the function timeCategorizing and adds it to the array
    timeCategoryArr.append(timeCategorizing(startTime))

# x train is implemented with the player lineup embedding and y train for the missing player
X_train = np.array([avgLineupEmbedding(players, playerEmbedding) for players in currentLineup])
y_train = np.array([playerEmbedding.wv[p] for p in missing_players if p in playerEmbedding.wv])

# converts the two arrays to numpy arrays to be used later
homeAwayBiasArr = np.array(homeAwayBiasArr)
timeCategoryArr = np.array(timeCategoryArr)

# combines all the vectors from the engineering features from above
X_train = np.hstack((X_train, homeAwayBiasArr, timeCategoryArr))

# converts x and y train to PyTorch tensors and assigns it to device to be used for cuda
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

# the pytorch model layout, 63 dimensions, 50 from lineup emb, 10 for home/away bias, 3 for time categorization
class NBAPlayerPredictionModel(nn.Module):
    def __init__(self, input_size=63, hidden_size=512, output_size=50):
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

# initialization of the model with the loss function being used
model = NBAPlayerPredictionModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CosineEmbeddingLoss()

# epoch of 15 seemed like the sweet spot without underfitting or overfitting
totalEpochs = 50
batch_size = 256

# moves the dataset over to tensor
dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor, torch.ones(len(y_train_tensor)))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# parameters for the epoch and how the batches are processed
for epoch in range(totalEpochs):
    total_loss = 0
    for batch_X, batch_y, target in dataloader:
        batch_X, batch_y, target = batch_X.to(device), batch_y.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"epoch [{epoch + 1} out of {totalEpochs}], loss: {total_loss / len(dataloader):.6f}")

# saves the trained model
torch.save(model.state_dict(), "nba_lineup_model.pth")
print("model saved")