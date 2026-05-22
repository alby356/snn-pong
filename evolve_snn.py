import numpy as np
import matplotlib.pyplot as plt

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

POP_SIZE = 50
GENERATIONS = 100
ELITE_FRAC = 0.2
MUTATION_RATE = 0.1
MUTATION_STRENGTH = 0.5

def encode_input(ball_y, paddle_center):
    ball_norm = ball_y / HEIGHT
    paddle_norm = paddle_center / HEIGHT
    diff_norm = (ball_y - paddle_center) / HEIGHT
    centers = np.linspace(0, 1, N_INPUT - 2)
    sigma = 1.0 / (N_INPUT - 2)
    ball_enc = np.exp(-0.5 * ((ball_norm - centers) / sigma) ** 2)
    return np.append(ball_enc, [diff_norm, paddle_norm]) * 3.0

def run_snn(W1, W2, max_steps=3000):
    ball_x = WIDTH // 2
    ball_y = np.random.randint(100, HEIGHT - 100)
    ball_dx = 3
    ball_dy = np.random.choice([-2, 2])

    ai_paddle_y = HEIGHT // 2 - paddle_height // 2
    left_paddle_y = HEIGHT // 2 - paddle_height // 2

    v_hidden = np.full(N_HIDDEN, -70.0)
    hits = 0
    misses = 0

    for step in range(max_steps):
        left_center = left_paddle_y + paddle_height // 2
        if left_center < ball_y and left_paddle_y < HEIGHT - paddle_height:
            left_paddle_y += paddle_speed
        elif left_center > ball_y and left_paddle_y > 0:
            left_paddle_y -= paddle_speed

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
                left_paddle_y <= ball_y <= left_paddle_y + paddle_height):
            ball_dx = abs(ball_dx)

        if (ball_x >= WIDTH - 30 - paddle_width - ball_size and
                ai_paddle_y <= ball_y <= ai_paddle_y + paddle_height):
            ball_dx = -abs(ball_dx)
            hits += 1

        if ball_x > WIDTH:
            misses += 1
            ball_x = WIDTH // 2
            ball_y = np.random.randint(100, HEIGHT - 100)
            ball_dx = abs(ball_dx)
            ball_dy = np.random.choice([-2, 2])

        if ball_x < 0:
            ball_x = WIDTH // 2
            ball_y = np.random.randint(100, HEIGHT - 100)
            ball_dx = abs(ball_dx)
            ball_dy = np.random.choice([-2, 2])

    fitness = hits - 0.5 * misses
    return fitness

def create_individual():
    W1 = np.random.randn(N_HIDDEN, N_INPUT) * 0.5
    W2 = np.random.randn(1, N_HIDDEN) * 0.5
    return (W1, W2)

def mutate(W1, W2):
    W1_new = W1.copy()
    W2_new = W2.copy()
    mask1 = np.random.random(W1.shape) < MUTATION_RATE
    mask2 = np.random.random(W2.shape) < MUTATION_RATE
    W1_new[mask1] += np.random.randn(mask1.sum()) * MUTATION_STRENGTH
    W2_new[mask2] += np.random.randn(mask2.sum()) * MUTATION_STRENGTH
    return W1_new, W2_new

def crossover(W1a, W2a, W1b, W2b):
    alpha = np.random.random()
    W1_child = alpha * W1a + (1 - alpha) * W1b
    W2_child = alpha * W2a + (1 - alpha) * W2b
    return W1_child, W2_child

print("Initializing population...")
population = [create_individual() for _ in range(POP_SIZE)]

best_fitness_history = []
avg_fitness_history = []
best_W1, best_W2 = None, None
best_ever = -np.inf

print(f"Evolving {POP_SIZE} SNNs for {GENERATIONS} generations...")

for gen in range(GENERATIONS):
    fitnesses = []
    for ind in population:
        W1, W2 = ind
        f = run_snn(W1, W2)
        fitnesses.append(f)

    fitnesses = np.array(fitnesses)

    best_idx = np.argmax(fitnesses)
    best_gen = fitnesses[best_idx]
    avg_gen = fitnesses.mean()

    best_fitness_history.append(best_gen)
    avg_fitness_history.append(avg_gen)

    if best_gen > best_ever:
        best_ever = best_gen
        best_W1, best_W2 = population[best_idx]

    print(f"Gen {gen:3d} | Best: {best_gen:.1f} | Avg: {avg_gen:.1f} | All-time best: {best_ever:.1f}")

    n_elite = int(POP_SIZE * ELITE_FRAC)
    elite_idx = np.argsort(fitnesses)[-n_elite:]
    elites = [population[i] for i in elite_idx]

    new_population = list(elites)

    while len(new_population) < POP_SIZE:
        p1 = elites[np.random.randint(len(elites))]
        p2 = elites[np.random.randint(len(elites))]
        p1_W1, p1_W2 = p1
        p2_W1, p2_W2 = p2
        child_W1, child_W2 = crossover(p1_W1, p1_W2, p2_W1, p2_W2)
        child_W1, child_W2 = mutate(child_W1, child_W2)
        new_population.append((child_W1, child_W2))

    population = new_population

print(f"\nEvolution complete. Best fitness: {best_ever:.1f}")

plt.figure(figsize=(12, 5))
plt.plot(best_fitness_history, label="Best fitness", linewidth=2)
plt.plot(avg_fitness_history, label="Avg fitness", linewidth=2, alpha=0.7)
plt.xlabel("Generation")
plt.ylabel("Fitness (hits - 0.5*misses)")
plt.title("SNN Evolution Progress")
plt.legend()
plt.show()

np.save("best_W1.npy", best_W1)
np.save("best_W2.npy", best_W2)
print("Best weights saved. Run pong_evolved.py to watch it play!")