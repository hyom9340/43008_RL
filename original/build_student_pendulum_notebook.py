from pathlib import Path

import nbformat as nbf


output_path = Path(__file__).resolve().parent / "Pendulum_Q_Learning_Student.ipynb"
notebook = nbf.v4.new_notebook()
notebook["metadata"] = {
    "kernelspec": {
        "display_name": "RL Pendulum (.venv)",
        "language": "python",
        "name": "rl-pendulum",
    },
    "language_info": {"name": "python", "version": "3.12"},
}

notebook["cells"] = [
    nbf.v4.new_markdown_cell("""# Pendulum-v1 using Q-Learning

**Name:** Replace with your name  
**Student ID:** Replace with your student ID

This experiment applies a Temporal Difference method (Q-Learning) to `Pendulum-v1`. The continuous state and action spaces are discretised so they can be represented by a Q-table. The agent is trained for 10,000 episodes.
"""),

    nbf.v4.new_markdown_cell(r"""## Method

The observation is $[\cos(\theta), \sin(\theta), \dot{\theta}]$ and the action is a torque between -2 and 2. The Q-Learning update is:

$$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma\max_{a'}Q(s',a') - Q(s,a)]$$

The agent uses epsilon-greedy exploration during training. A higher episode return is better because Pendulum rewards are negative costs.
"""),

    nbf.v4.new_markdown_cell("## 1. Imports and parameters"),

    nbf.v4.new_code_cell("""# Uncomment if the packages are not installed.
# %pip install \"gymnasium[classic-control]\" numpy matplotlib moviepy imageio-ffmpeg

from pathlib import Path
import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.wrappers import RecordVideo

SEED = 2026
TRAINING_EPISODES = 10_000
EVALUATION_EPISODES = 100

ANGLE_BINS = 36
VELOCITY_BINS = 36
ACTION_BINS = 15

GAMMA = 0.99
ALPHA = 0.18
MIN_ALPHA = 0.03
ALPHA_DECAY = 0.999

EPSILON = 1.0
MIN_EPSILON = 0.03
EPSILON_DECAY = 0.9975

np.random.seed(SEED)

OUTPUT_DIR = Path(\"pendulum_student_outputs\")
OUTPUT_DIR.mkdir(exist_ok=True)
"""),

    nbf.v4.new_markdown_cell("""## 2. Discretise the state and action spaces

The angle and angular velocity are each divided into 36 bins. The continuous torque is replaced by 15 evenly spaced actions.
"""),

    nbf.v4.new_code_cell("""angle_edges = np.linspace(-np.pi, np.pi, ANGLE_BINS + 1)[1:-1]
velocity_edges = np.linspace(-8.0, 8.0, VELOCITY_BINS + 1)[1:-1]
action_values = np.linspace(-2.0, 2.0, ACTION_BINS, dtype=np.float32)


def discretise_state(observation):
    cos_theta, sin_theta, angular_velocity = observation
    angle = np.arctan2(sin_theta, cos_theta)
    angle_index = np.digitize(angle, angle_edges)
    velocity_index = np.digitize(
        np.clip(angular_velocity, -8.0, 8.0),
        velocity_edges,
    )
    return int(angle_index), int(velocity_index)


q_table = np.zeros((ANGLE_BINS, VELOCITY_BINS, ACTION_BINS), dtype=np.float32)
print(\"Q-table shape:\", q_table.shape)
"""),

    nbf.v4.new_markdown_cell("## 3. Train the Q-Learning agent"),

    nbf.v4.new_code_cell("""env = gym.make(\"Pendulum-v1\")
rng = np.random.default_rng(SEED)

training_returns = []
epsilon = EPSILON
alpha = ALPHA

for episode in range(TRAINING_EPISODES):
    observation, info = env.reset(seed=SEED + episode)
    state = discretise_state(observation)
    terminated = False
    truncated = False
    episode_return = 0.0

    while not (terminated or truncated):
        # Epsilon-greedy action selection
        if rng.random() < epsilon:
            action_index = int(rng.integers(ACTION_BINS))
        else:
            action_index = int(np.argmax(q_table[state]))

        action = np.array([action_values[action_index]], dtype=np.float32)
        next_observation, reward, terminated, truncated, info = env.step(action)
        next_state = discretise_state(next_observation)

        # Q-Learning TD update
        future_q = 0.0 if terminated else np.max(q_table[next_state])
        td_target = reward + GAMMA * future_q
        td_error = td_target - q_table[state + (action_index,)]
        q_table[state + (action_index,)] += alpha * td_error

        state = next_state
        episode_return += reward

    training_returns.append(episode_return)
    epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)
    alpha = max(MIN_ALPHA, alpha * ALPHA_DECAY)

    if (episode + 1) % 500 == 0:
        recent_average = np.mean(training_returns[-100:])
        print(
            f\"Episode {episode + 1}/{TRAINING_EPISODES}, \"
            f\"average return (last 100) = {recent_average:.2f}\"
        )

env.close()
np.save(OUTPUT_DIR / \"pendulum_q_table.npy\", q_table)
print(\"Training completed.\")
"""),

    nbf.v4.new_markdown_cell("## 4. Training performance"),

    nbf.v4.new_code_cell("""window = 100
moving_average = np.convolve(
    training_returns,
    np.ones(window) / window,
    mode=\"valid\",
)

plt.figure(figsize=(10, 5))
plt.plot(training_returns, alpha=0.18, label=\"Episode return\")
plt.plot(
    np.arange(window - 1, TRAINING_EPISODES),
    moving_average,
    linewidth=2,
    label=\"100-episode moving average\",
)
plt.xlabel(\"Episode\")
plt.ylabel(\"Return (higher is better)\")
plt.title(\"Pendulum Q-Learning Training\")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / \"training_curve.png\", dpi=160)
plt.show()
"""),

    nbf.v4.new_markdown_cell("""## 5. Evaluate the trained policy

The trained greedy policy is evaluated without exploration. A random policy is included as the minimum comparison required to show that learning occurred.
"""),

    nbf.v4.new_code_cell("""def evaluate(use_trained_policy):
    evaluation_env = gym.make(\"Pendulum-v1\")
    returns = []

    for episode in range(EVALUATION_EPISODES):
        observation, info = evaluation_env.reset(seed=30_000 + episode)
        evaluation_env.action_space.seed(30_000 + episode)
        terminated = False
        truncated = False
        total_reward = 0.0

        while not (terminated or truncated):
            if use_trained_policy:
                state = discretise_state(observation)
                action_index = int(np.argmax(q_table[state]))
                action = np.array([action_values[action_index]], dtype=np.float32)
            else:
                action = evaluation_env.action_space.sample()

            observation, reward, terminated, truncated, info = evaluation_env.step(action)
            total_reward += reward

        returns.append(total_reward)

    evaluation_env.close()
    return np.array(returns)


trained_returns = evaluate(use_trained_policy=True)
random_returns = evaluate(use_trained_policy=False)

print(f\"Trained policy: {trained_returns.mean():.2f} ± {trained_returns.std():.2f}\")
print(f\"Random policy:  {random_returns.mean():.2f} ± {random_returns.std():.2f}\")
print(f\"Improvement:    {trained_returns.mean() - random_returns.mean():.2f}\")
"""),

    nbf.v4.new_markdown_cell("## 6. Record the trained agent"),

    nbf.v4.new_code_cell("""os.environ.setdefault(\"SDL_VIDEODRIVER\", \"dummy\")

video_env = gym.make(\"Pendulum-v1\", render_mode=\"rgb_array\")
video_env = RecordVideo(
    video_env,
    video_folder=str(OUTPUT_DIR / \"video\"),
    episode_trigger=lambda episode_id: True,
    name_prefix=\"pendulum-q-learning\",
    disable_logger=True,
)

observation, info = video_env.reset(seed=40_000)
terminated = False
truncated = False
video_return = 0.0

while not (terminated or truncated):
    state = discretise_state(observation)
    action_index = int(np.argmax(q_table[state]))
    action = np.array([action_values[action_index]], dtype=np.float32)
    observation, reward, terminated, truncated, info = video_env.step(action)
    video_return += reward

video_env.close()
print(f\"Recorded episode return: {video_return:.2f}\")
print(\"Video saved in:\", (OUTPUT_DIR / \"video\").resolve())
"""),

    nbf.v4.new_markdown_cell("""## 7. Short discussion

After running the notebook, report:

- the final 100-episode training average;
- the trained-policy mean and standard deviation;
- the random-policy mean and standard deviation;
- whether the trained agent performed better than random;
- whether the learning curve became stable;
- the limitation caused by discretising continuous states and torques.
"""),
]

nbf.validate(notebook)
nbf.write(notebook, output_path)
print(f"Created {output_path}")
