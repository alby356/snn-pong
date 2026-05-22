import numpy as np
import matplotlib.pyplot as plt

WIDTH, HEIGHT = 800, 600
ball_size = 10
paddle_height = 150
paddle_width = 10
paddle_speed = 5

dt = 1.0
threshold = -55
reset = -70
leak = 0.1

N_INPUT = 20
N_HIDDEN = 20
N_OUTPUT = 1

max_weight = 5.0
min_weight = 0.0
lr = 0.005
trace_decay = 0.95
reward_decay = 0.8

def make_population_voltages(n):
    return np.full(n, -70.0)

def encode_input(value, n_neurons, max_val=600):
    centers = np.linspace(0, max_val, n_neurons)
    sigma = max_val / n_neurons * 2
    rates = np.exp(-0.5 * ((value - centers) / sigma) ** 2)
    return rates * 3.0

def update_population(voltages, traces, input_currents):
    voltages += dt * (-leak * (voltages - reset) + input_currents)
    fired = voltages >= threshold
    voltages[fired] = reset
    traces *= trace_decay
    traces[fired] += 1.0
    return fired, voltages, traces

def run_episode(W1, W2, max_steps=5000):
    ball_x = WIDTH // 2
    ball_y = np.random.randint(100, HEIGHT - 100)
    ball_dx = 2
    ball_dy = np.random.choice([-1, 1])
    
    ai_paddle_y = HEIGHT // 2 - paddle_height // 2
    left_paddle_y = HEIGHT // 2 - paddle_height // 2
    
    v_hidden_up = make_population_voltages(N_HIDDEN)
    v_hidden_down = make_population_voltages(N_HIDDEN)
    v_out_up = np.array([-70.0])
    v_out_down = np.array([-70.0])
    
    traces_hidden_up = np.zeros(N_HIDDEN)
    traces_hidden_down = np.zeros(N_HIDDEN)
    traces_out_up = np.zeros(1)
    traces_out_down = np.zeros(1)
    
    reward_up = 0.0
    reward_down = 0.0
    last_action_up = False
    hits = 0
    
    for step in range(max_steps):
        left_center = left_paddle_y + paddle_height // 2
        if left_center < ball_y and left_paddle_y < HEIGHT - paddle_height:
            left_paddle_y += paddle_speed
        elif left_center > ball_y and left_paddle_y > 0:
            left_paddle_y -= paddle_speed
        
        paddle_center = ai_paddle_y + paddle_height // 2
        
        ball_input = encode_input(ball_y, N_INPUT)
        
        diff = paddle_center - ball_y
        up_scale = max(0, diff) / 300
        down_scale = max(0, -diff) / 300
        
        input_up = ball_input * up_scale
        input_down = ball_input * down_scale
        
        reward_up *= reward_decay
        reward_down *= reward_decay
        
        hidden_input_up = W1.T @ input_up
        hidden_input_down = W1.T @ input_down
        
        fired_hidden_up, v_hidden_up, traces_hidden_up = update_population(
            v_hidden_up, traces_hidden_up, hidden_input_up)
        fired_hidden_down, v_hidden_down, traces_hidden_down = update_population(
            v_hidden_down, traces_hidden_down, hidden_input_down)
        
        out_input_up = W2 @ fired_hidden_up.astype(float)
        out_input_down = W2 @ fired_hidden_down.astype(float)
        
        v_out_up += dt * (-leak * (v_out_up - reset) + out_input_up)
        v_out_down += dt * (-leak * (v_out_down - reset) + out_input_down)
        
        fired_out_up = v_out_up >= threshold
        fired_out_down = v_out_down >= threshold
        
        traces_out_up *= trace_decay
        traces_out_down *= trace_decay
        if fired_out_up[0]:
            v_out_up[0] = reset
            traces_out_up[0] += 1.0
        if fired_out_down[0]:
            v_out_down[0] = reset
            traces_out_down[0] += 1.0
        
        if reward_up != 0:
            W1 += lr * reward_up * np.outer(input_up, traces_hidden_up)
            W2 += lr * reward_up * np.outer(traces_out_up, traces_hidden_up)
            W1 = np.clip(W1, min_weight, max_weight)
            W2 = np.clip(W2, min_weight, max_weight)
        
        if reward_down != 0:
            W1 += lr * reward_down * np.outer(input_down, traces_hidden_down)
            W2 += lr * reward_down * np.outer(traces_out_down, traces_hidden_down)
            W1 = np.clip(W1, min_weight, max_weight)
            W2 = np.clip(W2, min_weight, max_weight)
        
        if fired_out_up[0] and ai_paddle_y > 0:
            ai_paddle_y -= paddle_speed
            last_action_up = True
        if fired_out_down[0] and ai_paddle_y < HEIGHT - paddle_height:
            ai_paddle_y += paddle_speed
            last_action_up = False
        
        ball_x += ball_dx
        ball_y += ball_dy
        
        if ball_y <= 0 or ball_y >= HEIGHT - ball_size:
            ball_dy *= -1
        
        if (ball_x <= 30 + paddle_width and
            left_paddle_y <= ball_y <= left_paddle_y + paddle_height):
            ball_dx = 2
        
        if (ball_x >= WIDTH - 30 - paddle_width - ball_size and
            ai_paddle_y <= ball_y <= ai_paddle_y + paddle_height):
            ball_dx = -2
            hits += 1
            if last_action_up:
                reward_up = 1.0
            else:
                reward_down = 1.0
        
        if ball_x > WIDTH:
            if last_action_up:
                reward_up = -1.0
            else:
                reward_down = -1.0
            return hits, False
        
        if ball_x < 0:
            ball_x = WIDTH // 2
            ball_y = np.random.randint(100, HEIGHT - 100)
            ball_dx = 2
            ball_dy = np.random.choice([-1, 1])
    
    return hits, True

W1 = np.random.uniform(0.1, 0.5, (N_INPUT, N_HIDDEN))
W2 = np.random.uniform(0.1, 0.5, (N_OUTPUT, N_HIDDEN))

print("Training expanded SNN with population coding...")

episodes = 2000
hit_history = []
window = 50

for ep in range(episodes):
    hits, survived = run_episode(W1, W2)
    hit_history.append(hits)
    
    if ep % 100 == 0:
        recent_avg = np.mean(hit_history[-window:]) if len(hit_history) >= window else np.mean(hit_history)
        print(f"Episode {ep} | Avg hits: {recent_avg:.2f} | W1_mean: {W1.mean():.3f} W2_mean: {W2.mean():.3f}")

print(f"\nFinal W1 mean: {W1.mean():.3f} W2 mean: {W2.mean():.3f}")

plt.figure(figsize=(10, 5))
plt.plot(hit_history, alpha=0.3, label="Hits per episode")

window_avg = []
for i in range(len(hit_history)):
    start = max(0, i - window)
    window_avg.append(np.mean(hit_history[start:i+1]))

plt.plot(window_avg, label=f"{window}-episode average", linewidth=2)
plt.xlabel("Episode")
plt.ylabel("Hits before missing")
plt.title("Expanded SNN - Population Coding + R-STDP")
plt.legend()
plt.show()