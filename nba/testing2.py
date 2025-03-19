# Group #8

# imports
import torch
import pandas as pd
import numpy as np
import pickle
import torch.nn as nn
from scipy.spatial.distance import cosine

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

# checks if cuda is usable if not use cpu
print("Cuda:", torch.__version__, torch.cuda.current_device(), torch.version.cuda, torch.cuda.get_device_name(0))

# use cuda or use cpu and lets user know which one is being used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initializes the model to the cuda/cpu device
model = NBAPlayerPredictionModel().to(device)

# loads the pretrained model
model.load_state_dict(torch.load("nba_lineup_model.pth"))

# evaluates with it
model.eval()

# importing all csv files to be used for and testing
df2007 = pd.read_csv("matchups-2007.csv")
df2008 = pd.read_csv("matchups-2008.csv")
df2009 = pd.read_csv("matchups-2009.csv")
df2010 = pd.read_csv("matchups-2010.csv")
df2011 = pd.read_csv("matchups-2011.csv")
df2012 = pd.read_csv("matchups-2012.csv")
df2013 = pd.read_csv("matchups-2013.csv")
df2014 = pd.read_csv("matchups-2014.csv")
df2015 = pd.read_csv("matchups-2015.csv")
test_df = pd.read_csv("NBA_test.csv")

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
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------

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

# ---------------------------------------------------------------------

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

# array to hold the all team members in their respected team
teamArray = {}

# function to sort players to their respected teams and stores it into the teamArray
for _, row in test_df.iterrows():
    home_team = row["home_team"]
    away_team = row["away_team"]

    for i in range(5):
        teamArray.setdefault(home_team, set()).add(row[f"home_{i}"])
        teamArray.setdefault(away_team, set()).add(row[f"away_{i}"])

# retreives the saved player embeddings from the training session from before
with open("playerEmbeddings.pkl", "rb") as f:
    playerVectors = pickle.load(f)

# setting up arrays to hold the data of each for the num of appearances for bias
homePlay = {}
awayPlay = {}

# goes through each game the play had played and counts home and away games
for _, row in test_df.iterrows():
    for i in range(5):
        player = row[f'home_{i}']
        homePlay[player] = homePlay.get(player, 0) + 1
        player = row[f'away_{i}']
        awayPlay[player] = awayPlay.get(player, 0) + 1

# function is used to calcualte a 2D vector for the requested player
def homeAwayBias(player):
    total_games = homePlay.get(player, 0) + awayPlay.get(player, 0)
    return [homePlay.get(player, 0) / total_games, awayPlay.get(player, 0) / total_games]

# function is used to calcualte the average distance for the word2vec embedding and returns it
def lineupEmbedding(players, embDictionary):
    possiblePlayer = [p for p in players if p in embDictionary]
    return np.mean([embDictionary[p] for p in possiblePlayer], axis=0) if possiblePlayer else np.zeros(50)

# takes the start minute and categorizes it
def startTimeCategorization(minute):
    # < 12 is early game, < 24 is mid-game, anything above is late game
    if minute < 12:
        return [1, 0, 0]
    elif minute < 24:
        return [0, 1, 0]
    else:
        return [0, 0, 1]

# find the most similar player for a given lineup
def closestPlayer(predEmbedded, team_name, possiblePlayers, embDictionary):
    bestPlayer = None
    highestSim = float("-inf")
    for possibleCandidate in possiblePlayers:
        if possibleCandidate in embDictionary:
            sim = 1 - cosine(predEmbedded, embDictionary[possibleCandidate])
            if sim > highestSim:
                highestSim = sim
                bestPlayer = possibleCandidate
    return bestPlayer


# array initialization
testLineups = []
testHomeTeams = []
unknown = []
row_indices = []
startTimes = []

# goes through each row in the test csv
for idx, row in test_df.iterrows():
    home_team = row["home_team"]
    knownPlayers = []
    missing = None

    # home_0 -> home_4 to find the missing spot
    for i in range(5):
        player = row[f"home_{i}"]
        if player == "?":
            missing = i
        else:
            knownPlayers.append(player)

    # runs all functions when missing is found
    if missing is not None:
        testLineups.append(knownPlayers)
        testHomeTeams.append(home_team)
        unknown.append(missing)
        row_indices.append(idx)
        startTimes.append(row["starting_min"] if "starting_min" in row else 0)

# prediction array to hold all predictions
predictions = []

# loop is used to loop through each row of the test csv
for lineup, home_team, missingSpot, row_idx, startTime in zip(testLineups, testHomeTeams, unknown, row_indices, startTimes):

    # calls function to retrieve lineup embed
    embeddedLineup = lineupEmbedding(lineup, playerVectors)

    # calls function to retreieve home/away bias
    homeBias = np.array([homeAwayBias(p) for p in lineup]).flatten()

    # if missing, assign default value that is neutral
    while len(homeBias) < 10:
        homeBias = np.append(homeBias, [0.5, 0.5])

    # calls function for the start time categorization
    startCategory = np.array(startTimeCategorization(startTime))

    # merges all the vectors from each function
    lineupTensor = torch.tensor(np.hstack((embeddedLineup, homeBias, startCategory)), dtype=torch.float32).unsqueeze(0).to(device)

    # predict the player embedding of unknown player
    with torch.no_grad():
        predictedEmb = model(lineupTensor).cpu().numpy().flatten()

    # grabs a list of all possible players from that season and 3 before it
    season = test_df.loc[row_idx, "season"]
    teamPlayers = possibleTeamPlayers(season, home_team)

    # best player compared with each other using cosine function
    bestPlayer = closestPlayer(predictedEmb, home_team, teamPlayers, playerVectors)

    predictions.append({"row_idx": row_idx, "season": season, "Fifth_Player": bestPlayer})

# saving the prediction via the dataframe and then to a csv file
predictions_df = pd.DataFrame(predictions)
predictions_df.to_csv("NBA predictions.csv", index=False)
print("predictions were saved into NBA predictions.csv")


# loading the true data given
labels_df = pd.read_csv("NBA_test_labels.csv")
labels_df.rename(columns={"removed_value": "True_Player"}, inplace=True)
predictions_df["True_Player"] = labels_df["True_Player"]
predictions_df["Correct"] = (predictions_df["Fifth_Player"] == predictions_df["True_Player"])

# calculates the average for the overall accuracy
overallAcc = predictions_df["Correct"].mean()
print(f"\noverall accuracy: {overallAcc:.2%}  ({predictions_df['Correct'].sum()} out of {len(predictions_df)})")

# separates the test data between seasons
if "season" in predictions_df.columns and predictions_df["season"].notnull().all():
    for seasonValue, group in predictions_df.groupby("season"):
        total = len(group)
        correct = group["Correct"].sum()
        accuracy = correct / total if total else 0
        print(f"\n{seasonValue} Test accuracy: {accuracy:.2%}")
        print(f"{int(correct)} out of {total} are correct")