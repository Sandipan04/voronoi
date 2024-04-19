import pygame 
import sys
import numpy as np
from datagen import *
import time
from voronoi_knn import *

# Initialize pygame
pygame.init()

# Constants for the game
size = 100
cell_size = 5
width, height = size*cell_size, size*cell_size
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Voronoi Game")

# Colors
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

def draw_grid(markers, red_points, blue_points, size=100):
    for x in range(size):
        for y in range(size):
            rect = pygame.Rect(x*cell_size, y*cell_size, cell_size, cell_size)
            if (x,y) in markers:
                pygame.draw.rect(screen, BLACK, rect)
            elif bool(red_points[x, y]) is True:
                pygame.draw.rect(screen, RED, rect)
            elif bool(blue_points[x, y]) is True:
                pygame.draw.rect(screen, BLUE, rect)
            # else:
            #     pygame.draw.rect(screen, BLACK, rect)
    pygame.display.update()

def human_vs_ai(model, data, size=100, cell_size=5, num_turns=5, quarantine_distance=5, start="model"):
    """Simulate a game and return the final grid and outcome."""
    grid = initialize_grid(size)
    markers = []
    current_state = []
    red_markers = []
    blue_markers = []
    player = 1 if start == "model" else 2
    moves = 0

    while moves < 2*num_turns:
        screen.fill(BLACK)
        
        if player == 1:
            action = model(current_state=current_state, data=data)
            x, y = divmod(action, size)
            valid_move = is_valid_move(grid, x, y, quarantine_distance, player)
            
            while not valid_move:
                x = random.randint(0, size - 1)
                y = random.randint(0, size - 1)
                valid_move = is_valid_move(grid, x, y, quarantine_distance, player)
            
            place_marker(grid, x, y, player)
            markers.append((x,y))
            current_state.append(size*x+y)
            red_markers.append((x,y))
            moves += 1
            player = 2
        
        else:
            event_list = pygame.event.get()
            for event in event_list:
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = pygame.mouse.get_pos()[0]//cell_size, pygame.mouse.get_pos()[1]//cell_size
                    valid_move = is_valid_move(grid, x, y, quarantine_distance, player)
                    if valid_move:
                        place_marker(grid, x, y, player)
                        markers.append((x,y))
                        current_state.append(size*x+y)
                        blue_markers.append((x,y))
                        moves += 1
                        player = 1
        
        red_points, blue_points = calculate_voronoi_points(grid, red_markers, blue_markers)
        draw_grid(markers=markers, red_points=red_points, blue_points=blue_points, size=size)
        pygame.display.flip()
        red_percentage, blue_percentage = calculate_area_percentage(red_points, blue_points)
        

    if red_percentage > blue_percentage:
        outcome = "Red wins"  # Red player wins
    elif red_percentage < blue_percentage:
        outcome = "Blue wins"  # Blue player wins
    else:
        outcome = "Tie"  # Tie

    return outcome, red_percentage, blue_percentage



if __name__ == "__main__":
    data = pd.read_csv("knn_data1_10.csv.csv")
    outcome, red_percentage, blue_percentage = human_vs_ai(model=model, data=data, size=size, cell_size=cell_size)
    # pygame.quit()
    # sys.exit()
    print(f"Result: {outcome}\nRed area: {red_percentage}%\nBlue area: {blue_percentage}%\n")
