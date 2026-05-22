import numpy as np
import pygame

WIDTH, HEIGHT = 800, 600
ball_size = 10
paddle_height = 80
paddle_width = 10
paddle_speed = 6

dt = 1.0
threshold = -55
reset = -70
leak = 0.1
N_INPUT = 10
N_HIDDEN = 20

def encode_input(ball_y, paddle_center):
    ball_norm = ball_y / HEIGHT
    paddle_norm = paddle_center / HEIGHT
    diff_norm = (ball_y - paddle_center) / HEIGHT
    centers = np.linspace(0, 1, N_INPUT - 2)
    sigma = 1.0 / (N_INPUT - 2)
    ball_enc = np.exp(-0.5 * ((ball_norm - centers) / sigma) ** 2)
    return np.append(ball_enc, [diff_norm, paddle_norm]) * 3.0

# Load best weights
W1 = np.load("best_W1.npy")
W2 = np.load("best_W2.npy")

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Evolved SNN Pong")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 74)
small_font = pygame.font.Font(None, 36)

ball_x, ball_y = WIDTH // 2, HEIGHT // 2
ball_dx, ball_dy = 3, 2

player_paddle_y = HEIGHT // 2 - paddle_height // 2
ai_paddle_y = HEIGHT // 2 - paddle_height // 2

v_hidden = np.full(N_HIDDEN, -70.0)

player_score = 0
ai_score = 0
hits = 0

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP] and player_paddle_y > 0:
        player_paddle_y -= paddle_speed
    if keys[pygame.K_DOWN] and player_paddle_y < HEIGHT - paddle_height:
        player_paddle_y += paddle_speed

    paddle_center = ai_paddle_y + paddle_height // 2
    inp = encode_input(ball_y, paddle_center)

    hidden_input = W1 @ inp
    v_hidden += dt * (-leak * (v_hidden - reset) + hidden_input)
    fired = v_hidden >= threshold
    v_hidden[fired] = reset

    output = (W2 @ fired.astype(float))[0]

    if output > 0.5 and ai_paddle_y < HEIGHT - paddle_height:
        ai_paddle_y += paddle_speed
    elif output < -0.5 and ai_paddle_y > 0:
        ai_paddle_y -= paddle_speed

    ball_x += ball_dx
    ball_y += ball_dy

    if ball_y <= 0 or ball_y >= HEIGHT - ball_size:
        ball_dy *= -1

    if (ball_x <= 30 + paddle_width and
            player_paddle_y <= ball_y <= player_paddle_y + paddle_height):
        ball_dx = abs(ball_dx)

    if (ball_x >= WIDTH - 30 - paddle_width - ball_size and
            ai_paddle_y <= ball_y <= ai_paddle_y + paddle_height):
        ball_dx = -abs(ball_dx)
        hits += 1

    if ball_x < 0:
        ai_score += 1
        ball_x, ball_y = WIDTH // 2, HEIGHT // 2
        ball_dx = 3
        ball_dy = np.random.choice([-2, 2])

    if ball_x > WIDTH:
        player_score += 1
        ball_x, ball_y = WIDTH // 2, HEIGHT // 2
        ball_dx = -3
        ball_dy = np.random.choice([-2, 2])

    screen.fill((0, 0, 0))
    pygame.draw.rect(screen, (255, 255, 255), (20, player_paddle_y, paddle_width, paddle_height))
    pygame.draw.rect(screen, (0, 255, 0), (WIDTH - 30, ai_paddle_y, paddle_width, paddle_height))
    pygame.draw.rect(screen, (255, 255, 255), (ball_x, ball_y, ball_size, ball_size))

    player_text = font.render(str(player_score), True, (255, 255, 255))
    ai_text = font.render(str(ai_score), True, (0, 255, 0))
    screen.blit(player_text, (WIDTH // 4, 20))
    screen.blit(ai_text, (3 * WIDTH // 4, 20))

    info_text = small_font.render(f"SNN Hits: {hits}", True, (200, 200, 200))
    screen.blit(info_text, (10, HEIGHT - 30))

    for y in range(0, HEIGHT, 20):
        pygame.draw.rect(screen, (255, 255, 255), (WIDTH // 2 - 2, y, 4, 10))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()