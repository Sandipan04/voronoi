import pygame
import sys
import numpy as np
import pandas as pd
import random
from datagen import initialize_grid, is_valid_move, place_marker, calculate_voronoi_points, calculate_area_percentage
from voronoi_knn import model

# Initialize Pygame
pygame.init()
pygame.font.init()  # Initialize font module

# Screen setup
size = 100
cell_size = 10
screen_width, screen_height = size*cell_size, size*cell_size
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Voronoi Game")

# Colors
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

class Button:
    def __init__(self, text, x, y, width, height, color, highlight_color, action):
        self.text = text
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.highlight_color = highlight_color
        self.action = action
        self.font = pygame.font.Font(None, 32)

    def draw(self, win):
        mouse_pos = pygame.mouse.get_pos()
        button_color = self.highlight_color if self.is_over(mouse_pos) else self.color
        pygame.draw.rect(win, button_color, (self.x, self.y, self.width, self.height))
        text_surf = self.font.render(self.text, True, (0, 0, 0))
        text_rect = text_surf.get_rect(center=(self.x + self.width // 2, self.y + self.height // 2))
        win.blit(text_surf, text_rect)

    def is_over(self, pos):
        return self.x <= pos[0] <= self.x + self.width and self.y <= pos[1] <= self.y + self.height

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.is_over(pygame.mouse.get_pos()):
                self.action()

def display_text(screen, text, position, color=(255, 255, 255)):
    """ Utility function to display text on screen """
    text_surface = pygame.font.Font(None, 36).render(text, True, color)
    text_rect = text_surface.get_rect(center=position)
    screen.blit(text_surface, text_rect)

def set_game_state(player):
    global game_state, start_player
    game_state = 'playing'
    start_player = player

def show_game_over(screen, outcome, red_percentage, blue_percentage):
    screen.fill(BLACK)  # Clear screen or set a suitable background
    display_text(screen, f"Result: {outcome}", (screen_width // 2, screen_height // 3), WHITE)
    display_text(screen, f"AI's area: {red_percentage}%", (screen_width // 2, screen_height // 2), RED)
    display_text(screen, f"Player's area: {blue_percentage}%", (screen_width // 2, 2 * screen_height // 3), BLUE)
    pygame.display.update()  # Update the display to show the text
    pygame.time.wait(5000)  # Wait for 5 seconds


button_width = 200
button_height = 50
x_human, x_ai = size*cell_size//2 - button_width//2, size*cell_size//2 - button_width//2
y_human, y_ai = size*cell_size//2 - 3*button_height//2, size*cell_size//2 + button_height//2
button_human = Button("Human Starts", x_human, y_human, button_width, button_height, RED, (200, 0, 0), lambda: set_game_state('human'))
button_ai = Button("AI Starts", x_ai, y_ai, button_width, button_height, BLUE, (0, 0, 200), lambda: set_game_state('ai'))

game_state = 'menu'
start_player = 'ai'
data = pd.read_csv("datasets/combined.csv")

def draw_grid(markers, red_points, blue_points, size=100, cell_size=10):
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

def game_loop():
    global game_state, start_player, size, cell_size
    num_turns = 5
    quarantine_distance = 5

    # Initialize game state
    grid = initialize_grid(size)
    markers = []
    red_markers = []
    blue_markers = []
    player = 1 if start_player == 'ai' else 2  # Determine who starts based on selection
    current_state = []
    moves = 0
    red_points, blue_points = calculate_voronoi_points(grid, red_markers, blue_markers)

    while game_state == 'playing' and moves < 2*num_turns:
        if player == 1:  # AI's turn
            action = model(current_state=current_state, data=data, red_points=red_points)
            x, y = divmod(action, size)
            valid_move = is_valid_move(grid, x, y, quarantine_distance, player)
            
            while not valid_move:
                x = random.randint(0, size - 1)
                y = random.randint(0, size - 1)
                valid_move = is_valid_move(grid, x, y, quarantine_distance, player)
            
            # print(x, y)
            place_marker(grid, x, y, player)
            markers.append((x,y))
            current_state.append(size*x+y)
            red_markers.append((x,y))
            moves += 1
            player = 2
        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game_state = 'menu'  # Return to menu or exit
                    return
                elif event.type == pygame.MOUSEBUTTONDOWN: # Human's turn
                    x, y = pygame.mouse.get_pos()
                    x //= cell_size
                    y //= cell_size
                    if is_valid_move(grid, x, y, quarantine_distance, player):
                        place_marker(grid, x, y, player)
                        markers.append((x, y))
                        blue_markers.append((x, y))
                        current_state.append(size * x + y)
                        player = 1  # Switch to AI
                        moves += 1

        red_points, blue_points = calculate_voronoi_points(grid, red_markers, blue_markers)
        draw_grid(markers, red_points, blue_points, size, cell_size)
        pygame.display.flip()

    pygame.time.wait(2000)
    red_percentage, blue_percentage = calculate_area_percentage(red_points, blue_points)

    game_state = 'menu'

    # After game logic determines the game is over
    outcome = "AI wins" if red_percentage > blue_percentage else "Player wins" if red_percentage < blue_percentage else "Tie"
    show_game_over(screen, outcome, red_percentage, blue_percentage)

    # pygame.time.wait(5000)  # Wait 5 seconds before closing
    return

    # After game loop
    print("Exiting game loop")

if __name__ == "__main__":
    # Main event loop
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif game_state == 'menu':
                button_human.handle_event(event)
                button_ai.handle_event(event)

        screen.fill(BLACK)  # Clear the screen
        if game_state == 'menu':
            button_human.draw(screen)
            button_ai.draw(screen)
        elif game_state == 'playing':
            game_loop()  # Enter the game loop if the state is 'playing'
        
        pygame.display.flip()  # Update the display
