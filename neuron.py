import numpy as np
import pygame

# Initialize pygame
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("SNN Pong")
clock = pygame.time.Clock()

# Pong objects
ball_x, ball_y = WIDTH // 2, HEIGHT // 2
ball_dx, ball_dy = 4, 3
ball_size = 10

paddle_height = 80
paddle_width = 10
paddle_speed = 5

# AI paddle (right side, controlled by SNN)
ai_paddle_y = HEIGHT // 2 - paddle_height // 2

# Player paddle (left side, controlled by keyboard)
player_paddle_y = HEIGHT // 2 - paddle_height // 2

# SNN parameters
dt = 1.0
threshold = -55
reset = -70
leak = 0.1
weights = [8.0, 8.0]

# Two networks - one for moving up, one for moving down
def make_neuron():
    return -70.0

v_up = [make_neuron(), make_neuron(), make_neuron()]
v_down = [make_neuron(), make_neuron(), make_neuron()]

def update_network(voltages, input_current):
    fired = [False, False, False]
    
    voltages[0] += dt * (-leak * (voltages[0] - reset) + input_current)
    if voltages[0] >= threshold:
        fired[0] = True
        voltages[0] = reset
    
    voltages[1] += dt * (-leak * (voltages[1] - reset))
    if fired[0]:
        voltages[1] += weights[0]
    if voltages[1] >= threshold:
        fired[1] = True
        voltages[1] = reset
    
    voltages[2] += dt * (-leak * (voltages[2] - reset))
    if fired[1]:
        voltages[2] += weights[1]
    if voltages[2] >= threshold:
        fired[2] = True
        voltages[2] = reset
    
    return fired[2]

# Scores
player_score = 0
ai_score = 0
font = pygame.font.Font(None, 74)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Player controls
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP] and player_paddle_y > 0:
        player_paddle_y -= paddle_speed
    if keys[pygame.K_DOWN] and player_paddle_y < HEIGHT - paddle_height:
        player_paddle_y += paddle_speed

    # SNN decides AI paddle movement
    # How far is ball above or below paddle center
    paddle_center = ai_paddle_y + paddle_height // 2
    diff = ball_y - paddle_center

    # Convert difference to input currents
    # Ball above paddle = move up signal
    # Ball below paddle = move down signal
    input_up = max(0, -diff) / 20
    input_down = max(0, diff) / 20

    fired_up = update_network(v_up, input_up)
    fired_down = update_network(v_down, input_down)

    # Move paddle based on which network fired
    if fired_up and ai_paddle_y > 0:
        ai_paddle_y -= paddle_speed
    if fired_down and ai_paddle_y < HEIGHT - paddle_height:
        ai_paddle_y += paddle_speed

    # Move ball
    ball_x += ball_dx
    ball_y += ball_dy

    # Ball bounces off top and bottom
    if ball_y <= 0 or ball_y >= HEIGHT - ball_size:
        ball_dy *= -1

    # Ball hits player paddle (left)
    if (ball_x <= 30 + paddle_width and
        player_paddle_y <= ball_y <= player_paddle_y + paddle_height):
        ball_dx *= -1

    # Ball hits AI paddle (right)
    if (ball_x >= WIDTH - 30 - paddle_width - ball_size and
        ai_paddle_y <= ball_y <= ai_paddle_y + paddle_height):
        ball_dx *= -1

    # Score
    if ball_x < 0:
        ai_score += 1
        ball_x, ball_y = WIDTH // 2, HEIGHT // 2
        ball_dx = 4
    if ball_x > WIDTH:
        player_score += 1
        ball_x, ball_y = WIDTH // 2, HEIGHT // 2
        ball_dx = -4

    # Draw
    screen.fill((0, 0, 0))
    
    # Paddles
    pygame.draw.rect(screen, (255, 255, 255), (20, player_paddle_y, paddle_width, paddle_height))
    pygame.draw.rect(screen, (255, 255, 255), (WIDTH - 30, ai_paddle_y, paddle_width, paddle_height))
    
    # Ball
    pygame.draw.rect(screen, (255, 255, 255), (ball_x, ball_y, ball_size, ball_size))
    
    # Scores
    player_text = font.render(str(player_score), True, (255, 255, 255))
    ai_text = font.render(str(ai_score), True, (255, 255, 255))
    screen.blit(player_text, (WIDTH // 4, 20))
    screen.blit(ai_text, (3 * WIDTH // 4, 20))
    
    # Net
    for y in range(0, HEIGHT, 20):
        pygame.draw.rect(screen, (255, 255, 255), (WIDTH // 2 - 2, y, 4, 10))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()