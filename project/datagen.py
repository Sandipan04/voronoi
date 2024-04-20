import numpy as np
import random
import pandas as pd
from tqdm import tqdm

def initialize_grid(size):
    """Initialize an empty grid."""
    grid = np.zeros((size, size, 3), dtype=int)
    return grid

def is_valid_move(grid, x, y, quarantine_distance, player):
    """Check if a move is valid based on the quarantine distance."""
    size = grid.shape[0]
    if x < 0 or x >= size or y < 0 or y >= size:
        return False
    if grid[x, y, 0] != 0 or grid[x, y, 1] != 0 or grid[x, y, 2] != 0:
        return False
    for i in range(size):
        for j in range(size):
            if grid[i, j, 0] != 0 and ((i - x)**2 + (j - y)**2)**(0.5) <= quarantine_distance:
                return False
            if grid[i, j, 1] != 0 and ((i - x)**2 + (j - y)**2)**(0.5) <= quarantine_distance:
                return False
    return True

def place_marker(grid, x, y, player):
    """Place a marker on the grid."""
    grid[x, y, player - 1] = 1

def calculate_voronoi_points(grid, red_markers, blue_markers):
    """Calculate the red and blue points based on the Voronoi diagram."""
    size = grid.shape[0]

    if len(blue_markers) == 0 and len(red_markers) == 0:
        red_markers = np.array(red_markers)
        empty_points = np.argwhere(np.all(grid == 0, axis=2))
        red_points = np.zeros((size, size), dtype=bool)
        blue_points = np.zeros((size, size), dtype=bool)
        red_points[empty_points[:, 0], empty_points[:, 1]] = False
        blue_points[empty_points[:, 0], empty_points[:, 1]] = False

    elif len(blue_markers) == 0:
        red_markers = np.array(red_markers)
        empty_points = np.argwhere(np.all(grid == 0, axis=2))
        red_points = np.zeros((size, size), dtype=bool)
        blue_points = np.zeros((size, size), dtype=bool)
        red_points[empty_points[:, 0], empty_points[:, 1]] = True
        blue_points[empty_points[:, 0], empty_points[:, 1]] = False

    elif len(red_markers) == 0:
        blue_markers = np.array(red_markers)
        empty_points = np.argwhere(np.all(grid == 0, axis=2))
        red_points = np.zeros((size, size), dtype=bool)
        blue_points = np.zeros((size, size), dtype=bool)
        red_points[empty_points[:, 0], empty_points[:, 1]] = False
        blue_points[empty_points[:, 0], empty_points[:, 1]] = True
        
    else:
        red_markers = np.array(red_markers)
        blue_markers = np.array(blue_markers)

        empty_points = np.argwhere(np.all(grid == 0, axis=2))

        red_dist = np.sqrt(np.sum((empty_points[:, np.newaxis, :] - red_markers[np.newaxis, :, :]) ** 2, axis=2)).min(axis=1)
        blue_dist = np.sqrt(np.sum((empty_points[:, np.newaxis, :] - blue_markers[np.newaxis, :, :]) ** 2, axis=2)).min(axis=1)

        red_points = np.zeros((size, size), dtype=bool)
        blue_points = np.zeros((size, size), dtype=bool)

        red_points[empty_points[:, 0], empty_points[:, 1]] = red_dist < blue_dist
        blue_points[empty_points[:, 0], empty_points[:, 1]] = blue_dist < red_dist

    return red_points, blue_points

def calculate_area_percentage(red_points, blue_points):
    """Calculate the area percentage occupied by red and blue points."""
    total_area = red_points.size
    red_area = np.sum(red_points)
    blue_area = np.sum(blue_points)
    red_percentage = (red_area / total_area) * 100
    blue_percentage = (blue_area / total_area) * 100
    return red_percentage, blue_percentage

def simulate_game(size=100, num_turns=5, quarantine_distance=5):
    """Simulate a game and return the final grid and outcome."""
    grid = initialize_grid(size)
    markers = []
    red_markers = []
    blue_markers = []
    player = 1  # Red player starts

    for _ in range(num_turns * 2):
        valid_move = False
        while not valid_move:
            x = random.randint(0, size - 1)
            y = random.randint(0, size - 1)
            valid_move = is_valid_move(grid, x, y, quarantine_distance, player)
        place_marker(grid, x, y, player)
        if player == 1:
            markers.append(size*x + y)
            red_markers.append((x,y))
        else:
            markers.append(size*x + y)
            blue_markers.append((x,y))
        player = 2 if player == 1 else 1  # Switch player

    red_points, blue_points = calculate_voronoi_points(grid, red_markers, blue_markers)
    red_percentage, blue_percentage = calculate_area_percentage(red_points, blue_points)

    return markers + [red_percentage, blue_percentage]

def generate_dataset(num_games, size=100, num_turns=5, quarantine_distance=5):
    """Generates a dataset of games."""
    columns = ['Move_{}_P{}'.format(i + 1, 1 + i % 2) for i in range(2 * num_turns)] + ['Area_P1', 'Area_P2']
    data = []
    for _ in tqdm(range(num_games), desc="Generating games"):
        game_data = simulate_game(size, num_turns, quarantine_distance)
        data.append(game_data)
    return pd.DataFrame(data, columns=columns)

if __name__ =="__main__":
    num_games = 100
    size = 100
    num_turns = 5
    quarantine_distance = 5

    # dataset = generate_dataset(num_games, size, num_turns, quarantine_distance)
    # dataset.to_csv('voronoi_data.csv', index=False)

    simulate_game()