import pygame 
import sys
import numpy as np
from datagen import *
import time
from voronoi_knn import *



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

def human_vs_ai(model, data, size=100, cell_size=10, num_turns=5, quarantine_distance=5, start="model"):
    """Simulate a game and return the final grid and outcome."""
    grid = initialize_grid(size)
    markers = []
    current_state = []
    red_markers = []
    blue_markers = []
    player = 1 if start == "model" else 2
    moves = 0
    red_points, blue_points = calculate_voronoi_points(grid, red_markers, blue_markers)

    while moves < 2*num_turns:
        screen.fill(BLACK)
        
        if player == 1:
            action = model(current_state=current_state, data=data, red_points=red_points)
            x, y = divmod(action, size)
            valid_move = is_valid_move(grid, x, y, quarantine_distance, player)
            
            while not valid_move:
                x = random.randint(0, size - 1)
                y = random.randint(0, size - 1)
                valid_move = is_valid_move(grid, x, y, quarantine_distance, player)
            
            print(x, y)
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
        outcome = "AI wins"  # Red player wins
    elif red_percentage < blue_percentage:
        outcome = "Player wins"  # Blue player wins
    else:
        outcome = "Tie"  # Tie

    if start == "model":
        return outcome, red_percentage, blue_percentage, current_state + [red_percentage, blue_percentage]
    else:
        return outcome, red_percentage, blue_percentage, current_state + [blue_percentage, red_percentage]

def game():
    while True:
        # Initialize pygame
        pygame.init()

        # Constants for the game
        size = 100
        cell_size = 10
        width, height = size*cell_size, size*cell_size
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Voronoi Game")

        n = str(input("Press 1 for player 1 and press 2 for player 2, any other key to quit: "))

        num_turns = 5
        columns = ['Move_{}_P{}'.format(i + 1, 1 + i % 2) for i in range(2 * num_turns)] + ['Area_P1', 'Area_P2']
        new_data = []

        if n == str(1):
            data = pd.read_csv(f"datasets/combined.csv")
            screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Voronoi Game")

            outcome, ai, human, game_data = human_vs_ai(model=model, data=data, size=size, cell_size=cell_size, start="human")
            time.sleep(30)
            print(f"Result: {outcome}\nAI's area: {ai}%\nPlayer's area: {human}%\n")
            
            new_data.append(game_data)
            # print(new_data)
            dataset = pd.DataFrame(new_data, columns=columns)
            dataset.to_csv("datasets/combined.csv", mode='a', index=False, header=False)

        elif n == str(2):
            data = pd.read_csv(f"datasets/combined.csv")
            screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Voronoi Game")
            outcome, ai, human, game_data = human_vs_ai(model=model, data=data, size=size, cell_size=cell_size, start="model")
            time.sleep(30)
            print(f"Result: {outcome}\nAI's area: {ai}%\nPlayer's area: {human}%\n")

            new_data.append(game_data)
            # print(new_data)
            dataset = pd.DataFrame(new_data, columns=columns)
            dataset.to_csv("datasets/combined.csv", mode='a', index=False, header=False)
        else:
            break

        pygame.quit()


if __name__ == "__main__":
    game()

