import numpy as np
from sklearn.neighbors import NearestNeighbors
import random
import pandas as pd
from tqdm import tqdm
from datagen import *
from voronoi_knn import *
import time

def model_vs_random(model, data, size=100, num_turns=5, quarantine_distance=5, start="model"):
    """Simulate a game and return the final grid and outcome."""
    grid = initialize_grid(size)
    markers = []
    red_markers = []
    blue_markers = []
    player = 1 if start == "model" else 2 

    for _ in range(num_turns * 2):
        
        if player == 1:
            action = model(current_state=markers, data=data)
            x, y = divmod(action, size)
            valid_move = is_valid_move(grid, x, y, quarantine_distance, player)
            
            while not valid_move:
                x = random.randint(0, size - 1)
                y = random.randint(0, size - 1)
                valid_move = is_valid_move(grid, x, y, quarantine_distance, player)
            
            place_marker(grid, x, y, player)
            markers.append(size*x+y)
            red_markers.append((x,y))
        
        else:
            valid_move = False
            
            while not valid_move:
                x = random.randint(0, size - 1)
                y = random.randint(0, size - 1)
                valid_move = is_valid_move(grid, x, y, quarantine_distance, player)
            
            place_marker(grid, x, y, player)
            markers.append(size*x+y)
            blue_markers.append((x,y))

        player = 2 if player == 1 else 1  # Switch player

    red_points, blue_points = calculate_voronoi_points(grid, red_markers, blue_markers)
    red_percentage, blue_percentage = calculate_area_percentage(red_points, blue_points)

    if red_percentage > blue_percentage:
        outcome = "model wins"  # Red player wins
    elif red_percentage < blue_percentage:
        outcome = "model loses"  # Blue player wins
    else:
        outcome = "tie"  # Tie

    return outcome, markers + [red_percentage, blue_percentage]

def simulate_model_vs_random(model, data, name_to_save, num_games=100, size=100, num_turns=5, quarantine_distance=5, start="model"):
    win_count = 0
    columns = ['Move_{}_P{}'.format(i + 1, 1 + i % 2) for i in range(2 * num_turns)] + ['Area_P1', 'Area_P2']
    new_data = []
    for _ in tqdm(range(num_games), desc="Games Played"):
        outcome, game_data = model_vs_random(model, data, size=100, num_turns=5, quarantine_distance=5, start="model")
        new_data.append(game_data)

        # print(outcome)
        if outcome == "model wins":
            win_count += 1
    
    dataset = pd.DataFrame(new_data, columns=columns)
    dataset.to_csv(name_to_save, index=False)
    win_percentage = (win_count/num_games)*100
    return win_percentage

if __name__ == "__main__":
    iterations = 10
    data = pd.read_csv("voronoi_data.csv")
    for i in range(iterations):
        print(f"Iteration: {i+1}")
        name = f"knn_data2_{i+1}.csv"
        tic = time.localtime()
        # print(tic)
        win_percentage = simulate_model_vs_random(model=model, data=data, name_to_save=name, num_games=10000, start="random")
        toc = time.localtime()
        with open("record.txt", 'a') as file:
            file.write(f"iteration: {i}, start time: {tic}, end time: {toc}\n")
        data = pd.read_csv(name)
        print(f"Win Percentage: {win_percentage}%")
