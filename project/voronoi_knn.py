from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np

def column_selector(t, column_headers):
    features = column_headers[:t]
    label = column_headers[t]
    area = column_headers[-1] if t%2==1 else column_headers[-2]
    return features, label, area

def model(current_state:list, data, red_points):
    t = len(current_state)
    if t == 0:
        y = data["Area_P1"] # outcome of the t+1-th player
        next_move_set = data["Move_1_P1"] # next move set

        indices = np.random.randint(0, data.shape[0], size=20)
    
    else: 
        column_headers = data.columns

        # Filtering data
        features, label, area = column_selector(t, column_headers)
        X = data[features] # features: first t columns
        y = data[str(area)] # outcome of the t+1-th player
        next_move_set = data[str(label)] # next move set

        # Initialize NearestNeighbors
        nn = NearestNeighbors(n_neighbors=20)
        nn.fit(X)

        # Ensure current_state is a DataFrame with correct feature names
        current_state_df = pd.DataFrame([current_state], columns=features)

        #  Find k nearest neighbors
        distances, indices = nn.kneighbors(current_state_df)
        indices = indices.flatten()

    #  Analyze the historical success of t+1-th move in these neighbors
    move_scores = {}
    size = 100
    for index in indices:
        move = next_move_set.iloc[index]
        area_control = y.iloc[index]

        a, b = divmod(move, size)
        if red_points[a][b] == True:
            area_control -= 10

        if move in move_scores:
            move_scores[move].append(area_control)
        else:
            move_scores[move] = [area_control]
    
    best_move = max(move_scores, key=lambda x: np.mean(move_scores[x]))  # select move with highest average outcome

    return best_move