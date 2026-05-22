import numpy as np
import matplotlib.pyplot as plt

WIDTH, HEIGHT = 800, 600
ball_size = 10
paddle_height = 80
paddle_width = 10
paddle_speed = 5

POP_SIZE = 100
GENERATIONS = 300
ELITE_FRAC = 0.2
MUTATION_RATE = 0.1
MUTATION_STRENGTH = 0.3
N_HIDDEN = 15

def run_game(weights, max_steps=5000):
    W1 = weights[:N_HIDDEN * 3].reshape(N_HIDDEN, 3)
    W2 = weights[N_HIDDEN * 3:].reshape(1, N_HIDDEN)
    
    ball_x = WIDTH // 2
    ball_y = np.random.randint(100, HEIGHT - 100)
    ball_dx = 3
    ball_dy = np.random.choice([-2, 2])
    
    ai_y = HEIGHT // 2 - paddle_height // 2
    left_y = HEIGHT // 2 - paddle_height // 2
    
    hits = 0
    misses = 0
    total_distance = 0
    
    for step in range(max_steps):
        # Left paddle follows ball
        left_center = left_y + paddle_height // 2
        if left_center < ball_y:
            left_y = min(left_y + paddle_speed, HEIGHT - paddle_height)
        else:
            left_y = max(left_y - paddle_speed, 0)
        
        # SNN input: just 3 numbers
        ai_center = ai_y + paddle_height // 2
        inp = np.array([
            (ball_y - ai_center) / HEIGHT,  # where is ball relative to paddle
            ball_dy / 5.0,                   # ball vertical speed
            (ball_x - WIDTH/2) / WIDTH       # is ball on our side
        ])
        
        # Simple forward pass with tanh activation
        hidden = np.tanh(W1 @ inp)
        output = np.tanh((W2 @ hidden)[0])
        
        # Move paddle
        ai_y += output * paddle_speed
        ai_y = max(0, min(ai_y, HEIGHT - paddle_height))
        
        # Track how close paddle stays to ball
        total_distance += abs(ball_y - ai_center)
        
        # Move ball
        ball_x += ball_dx
        ball_y += ball_dy
        
        if ball_y <= 0 or ball_y >= HEIGHT - ball_size:
            ball_dy *= -1
        
        if (ball_x <= 30 + paddle_width and
                left_y <= ball_y <= left_y + paddle_height):
            ball_dx = abs(ball_dx)
        
        if (ball_x >= WIDTH - 30 - paddle_width - ball_size and
                ai_y <= ball_y <= ai_y + paddle_height):
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
    
    # Fitness rewards hits and penalizes distance from ball
    avg_distance = total_distance / max_steps
    fitness = hits * 100 - misses * 20
    return fitness

# Total weights needed
n_weights = N_HIDDEN * 3 + N_HIDDEN

def create_individual():
    return np.random.randn(n_weights) * 0.5

def mutate(weights):
    new = weights.copy()
    mask = np.random.random(len(new)) < MUTATION_RATE
    new[mask] += np.random.randn(mask.sum()) * MUTATION_STRENGTH
    return new

def crossover(w1, w2):
    alpha = np.random.random(len(w1))
    return alpha * w1 + (1 - alpha) * w2

print("Initializing population...")
population = [create_individual() for _ in range(POP_SIZE)]

best_fitness_history = []
avg_fitness_history = []
best_weights = None
best_ever = -np.inf

print(f"Evolving {POP_SIZE} networks for {GENERATIONS} generations...")

for gen in range(GENERATIONS):
    fitnesses = np.array([run_game(w) for w in population])
    
    best_idx = np.argmax(fitnesses)
    best_gen = fitnesses[best_idx]
    avg_gen = fitnesses.mean()
    
    best_fitness_history.append(best_gen)
    avg_fitness_history.append(avg_gen)
    
    if best_gen > best_ever:
        best_ever = best_gen
        best_weights = population[best_idx].copy()
    
    if gen % 10 == 0:
        print(f"Gen {gen:3d} | Best: {best_gen:.1f} | Avg: {avg_gen:.1f} | All-time: {best_ever:.1f}")
    
    n_elite = int(POP_SIZE * ELITE_FRAC)
    elite_idx = np.argsort(fitnesses)[-n_elite:]
    elites = [population[i] for i in elite_idx]
    
    new_population = list(elites)
    
    while len(new_population) < POP_SIZE:
        p1 = elites[np.random.randint(len(elites))]
        p2 = elites[np.random.randint(len(elites))]
        child = crossover(p1, p2)
        child = mutate(child)
        new_population.append(child)
    
    population = new_population

print(f"\nEvolution complete. Best fitness: {best_ever:.1f}")

plt.figure(figsize=(12, 5))
plt.plot(best_fitness_history, label="Best fitness", linewidth=2)
plt.plot(avg_fitness_history, label="Avg fitness", linewidth=2, alpha=0.7)
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("SNN Evolution Progress")
plt.legend()
plt.show()

np.save("best_weights.npy", best_weights)
print("Saved. Run pong_evolved.py to play!")