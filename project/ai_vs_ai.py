import numpy as np
from sklearn.neighbors import NearestNeighbors
import random
import pandas as pd
from tqdm import tqdm
from datagen import *
from voronoi_knn import *

def model_vs_random(model, data1, data2, size=100, num_turns=5, quarantine_distance=5):
    """Simulate a game and return the final grid and outcome."""
    grid = initialize_grid(size)
    markers = []
    red_markers = []
    blue_markers = []
    player = 1

    for _ in range(num_turns * 2):
        
        if player == 1:
            action = model(current_state=markers, data=data1)
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
            action = model(current_state=markers, data=data2)
            x, y = divmod(action, size)
            valid_move = is_valid_move(grid, x, y, quarantine_distance, player)
            
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
        outcome = "red wins"  # Red player wins
    elif red_percentage < blue_percentage:
        outcome = "blue wins"  # Blue player wins
    else:
        outcome = "tie"  # Tie

    return outcome, markers + [red_percentage, blue_percentage]

def simulate_ai_vs_ai(model, data1, data2, name_to_save, num_games=10000, size=100, num_turns=5, quarantine_distance=5):
    model1 = 0
    model2 = 0
    columns = ['Move_{}_P{}'.format(i + 1, 1 + i % 2) for i in range(2 * num_turns)] + ['Area_P1', 'Area_P2']
    new_data = []
    for _ in tqdm(range(num_games), desc="Games Played"):
        outcome, game_data = model_vs_random(model, data1, data2, size=100, num_turns=5, quarantine_distance=5)
        new_data.append(game_data)
        if outcome == "red wins":
            model1 += 1
        elif outcome == "blue wins":
            model2 += 1
    
    model1_percentage = (model1/num_games)*100
    model2_percentage = (model2/num_games)*100
    tie_percenrtage = ((num_games-model1-model2)/num_games)*100
    dataset = pd.DataFrame(new_data, columns=columns)
    dataset.to_csv(name_to_save, index=False)
    return model1_percentage, model2_percentage, tie_percenrtage

if __name__ == "__main__":
    for i in range(30,31):
        print("--------------------")
        print(f"Iteration: {i}")
        data1, data2 = pd.read_csv("datasets/human1.csv"), pd.read_csv("datasets/human2.csv")
        name = f"datasets/new_data_{i}.csv"
        model1_percentage, model2_percentage, tie_percenrtage = simulate_ai_vs_ai(model=model, data1=data1, data2=data2, name_to_save=name, num_games=10000)
        print(f"Win Percentage based on area controlled: {model1_percentage}% : {model2_percentage}% : {tie_percenrtage}%")
        data1, data2 = pd.read_csv(name), pd.read_csv(name)
