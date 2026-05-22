import numpy as np
import pygame

WIDTH, HEIGHT = 800, 600
ball_size = 10
paddle_height = 80
paddle_width = 10
paddle_speed = 5
N_HIDDEN = 15

weights = np.load("best_weights.npy")
W1 = weights[:N_HIDDEN * 3].reshape(N_HIDDEN, 3)
W2 = weights[N_HIDDEN * 3:].reshape(1, N_HIDDEN)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Evolved SNN Pong")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 74)
small_font = pygame.font.Font(None, 36)

ball_x, ball_y = WIDTH // 2, HEIGHT // 2
ball_dx, ball_dy = 3, 2

player_paddle_y = HEIGHT // 2 - paddle_height // 2
ai_y = HEIGHT // 2 - paddle_height // 2

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

    ai_center = ai_y + paddle_height // 2
    inp = np.array([
        (ball_y - ai_center) / HEIGHT,
        ball_dy / 5.0,
        (ball_x - WIDTH/2) / WIDTH
    ])

    hidden = np.tanh(W1 @ inp)
    output = np.tanh((W2 @ hidden)[0])

    # Fixed speed movement like player
    if output > 0.1 and ai_y < HEIGHT - paddle_height:
        ai_y += paddle_speed
    elif output < -0.1 and ai_y > 0:
        ai_y -= paddle_speed

    ball_x += ball_dx
    ball_y += ball_dy

    if ball_y <= 0 or ball_y >= HEIGHT - ball_size:
        ball_dy *= -1

    # Player paddle hit
    if (ball_x <= 20 + paddle_width and
            ball_x >= 20 and
            player_paddle_y <= ball_y + ball_size and
            ball_y <= player_paddle_y + paddle_height):
        ball_dx = abs(ball_dx)
        ball_x = 20 + paddle_width + 1

    # AI paddle hit
    ai_paddle_x = WIDTH - 30
    if (ball_x + ball_size >= ai_paddle_x and
            ball_x <= ai_paddle_x + paddle_width and
            ai_y <= ball_y + ball_size and
            ball_y <= ai_y + paddle_height):
        ball_dx = -abs(ball_dx)
        ball_x = ai_paddle_x - ball_size - 1
        hits += 1

    # Player scores - ball goes past AI
    if ball_x + ball_size > WIDTH:
        player_score += 1
        ball_x, ball_y = WIDTH // 2, HEIGHT // 2
        ball_dx = -3
        ball_dy = np.random.choice([-2, 2])

    # AI scores - ball goes past player
    if ball_x < 0:
        ai_score += 1
        ball_x, ball_y = WIDTH // 2, HEIGHT // 2
        ball_dx = 3
        ball_dy = np.random.choice([-2, 2])

    screen.fill((0, 0, 0))
    pygame.draw.rect(screen, (255, 255, 255), (20, player_paddle_y, paddle_width, paddle_height))
    pygame.draw.rect(screen, (0, 255, 0), (ai_paddle_x, ai_y, paddle_width, paddle_height))
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