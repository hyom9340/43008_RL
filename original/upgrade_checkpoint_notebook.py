from pathlib import Path

import nbformat


root = Path(__file__).resolve().parent
source_path = root / "Pendulum_TD_Q_Learning.ipynb"
output_path = root / "Pendulum_TD_Q_Learning_Checkpoint.ipynb"

notebook = nbformat.read(source_path, as_version=4)
notebook.metadata["kernelspec"] = {
    "display_name": "Python (RL .venv)",
    "language": "python",
    "name": "rl-venv",
}

notebook.cells[0].source = notebook.cells[0].source.replace(
    "Pendulum-v1 with Discretised Q-Learning",
    "Pendulum-v1 with Checkpointed Discretised Q-Learning",
)

notebook.cells[9].source = '''# ---------------- CHECKPOINT TRAINING CONTROLS ----------------
# True: load the latest checkpoint and add more episodes.
# False: create a brand-new experiment. Use a new EXPERIMENT_NAME.
RESUME_TRAINING = True

# Desired cumulative total. With a 50,000 checkpoint this runs only 30,000 more.
TARGET_TOTAL_EPISODES = 80000

# Change this whenever bins, reward design, or core hyperparameters change.
EXPERIMENT_NAME = "pendulum_qlearning_36x36x15_seed2026_from50000"

CONFIG = {
    "environment": "Pendulum-v1",
    "target_total_episodes": TARGET_TOTAL_EPISODES,
    "evaluation_episodes": 100,
    "random_baseline_episodes": 100,
    "angle_bins": 36,
    "velocity_bins": 36,
    "action_bins": 15,
    "gamma": 0.99,
    "initial_alpha": 0.18,
    "minimum_alpha": 0.03,
    "alpha_decay": 0.999,
    "initial_epsilon": 1.0,
    "minimum_epsilon": 0.03,
    "epsilon_decay": 0.9975,
    "seed": SEED,
}

config_table = pd.DataFrame(
    {"Parameter": list(CONFIG.keys()), "Value": list(CONFIG.values())}
)
display(config_table)

with open(RESULT_DIR / f"{EXPERIMENT_NAME}_config.json", "w", encoding="utf-8") as file:
    json.dump(CONFIG, file, indent=2)
'''

notebook.cells[14].source = '''## 8. Checkpoint-aware Q-Learning training

With `RESUME_TRAINING=True`, the code loads the latest Q-table, completed episode count, epsilon, alpha, random-generator state and combined training history. It trains only the difference between `TARGET_TOTAL_EPISODES` and the completed count. A checkpoint is written only after the new block finishes successfully.

Set `RESUME_TRAINING=False` only for a new experiment name. This safety rule prevents accidentally overwriting an existing run.
'''

notebook.cells[15].source = '''def choose_epsilon_greedy_action(q_values, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(len(q_values)))
    best_actions = np.flatnonzero(q_values == np.max(q_values))
    return int(rng.choice(best_actions))


CHECKPOINT_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}_latest_checkpoint.npz"
HISTORY_PATH = RESULT_DIR / f"{EXPERIMENT_NAME}_training_history.csv"
BEST_MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}_best_q_table.npy"
BEST_METADATA_PATH = RESULT_DIR / f"{EXPERIMENT_NAME}_best_metadata.json"


def initialise_or_resume_training(config, resume_training):
    if resume_training and CHECKPOINT_PATH.exists():
        checkpoint = np.load(CHECKPOINT_PATH, allow_pickle=False)
        q_table = checkpoint["q_table"].astype(np.float32, copy=True)
        completed_episodes = int(checkpoint["completed_episodes"])
        epsilon = float(checkpoint["epsilon"])
        alpha = float(checkpoint["alpha"])
        rng = np.random.default_rng()
        rng.bit_generator.state = json.loads(str(checkpoint["rng_state"].item()))

        if q_table.shape != q_table_shape:
            raise ValueError(
                f"Checkpoint shape {q_table.shape} does not match {q_table_shape}. "
                "Use a new EXPERIMENT_NAME for changed discretisation."
            )
        if not HISTORY_PATH.exists():
            raise FileNotFoundError("Checkpoint exists but its history CSV is missing.")

        history = pd.read_csv(HISTORY_PATH)
        if len(history) != completed_episodes:
            raise ValueError("Checkpoint episode count and history length disagree.")

        print(f"Resuming after {completed_episodes} completed episodes.")
        print(f"Loaded epsilon={epsilon:.4f}, alpha={alpha:.4f}")
        return q_table, completed_episodes, epsilon, alpha, rng, history

    if resume_training and not CHECKPOINT_PATH.exists():
        print("No checkpoint found; starting the first block of this experiment.")

    if not resume_training and CHECKPOINT_PATH.exists():
        raise FileExistsError(
            "This EXPERIMENT_NAME already has a checkpoint. Choose a new "
            "EXPERIMENT_NAME before starting a fresh experiment."
        )

    q_table = np.zeros(q_table_shape, dtype=np.float32)
    rng = np.random.default_rng(config["seed"])
    return (
        q_table,
        0,
        config["initial_epsilon"],
        config["initial_alpha"],
        rng,
        pd.DataFrame(),
    )


def save_latest_checkpoint(q_table, completed_episodes, epsilon, alpha, rng):
    np.savez_compressed(
        CHECKPOINT_PATH,
        q_table=q_table,
        completed_episodes=np.int64(completed_episodes),
        epsilon=np.float64(epsilon),
        alpha=np.float64(alpha),
        rng_state=np.array(json.dumps(rng.bit_generator.state)),
    )


def train_q_learning(config, resume_training=True):
    env = gym.make(config["environment"])
    (
        q_table,
        completed_episodes,
        epsilon,
        alpha,
        rng,
        old_history,
    ) = initialise_or_resume_training(config, resume_training)

    new_records = []
    start_episode = completed_episodes
    end_episode = max(completed_episodes, config["target_total_episodes"])
    episodes_to_run = end_episode - completed_episodes
    print(f"Target total episodes: {config['target_total_episodes']}")
    print(f"Episodes required in this run: {episodes_to_run}")

    for episode_index in range(start_episode, end_episode):
        observation, info = env.reset(seed=config["seed"] + episode_index)
        state = observation_to_state(observation)
        terminated = truncated = False
        episode_return = 0.0
        absolute_td_errors = []
        episode_steps = 0

        while not (terminated or truncated):
            action_index = choose_epsilon_greedy_action(q_table[state], epsilon, rng)
            env_action = action_index_to_env_action(action_index)
            next_observation, reward, terminated, truncated, info = env.step(env_action)
            next_state = observation_to_state(next_observation)

            # Time-limit truncation is not a natural terminal state, so bootstrap through it.
            future_value = 0.0 if terminated else float(np.max(q_table[next_state]))
            td_target = reward + config["gamma"] * future_value
            td_error = td_target - q_table[state + (action_index,)]
            q_table[state + (action_index,)] += alpha * td_error

            state = next_state
            episode_return += reward
            absolute_td_errors.append(abs(float(td_error)))
            episode_steps += 1

        new_records.append(
            {
                "episode": episode_index + 1,
                "return": episode_return,
                "epsilon": epsilon,
                "alpha": alpha,
                "mean_absolute_td_error": np.mean(absolute_td_errors),
                "episode_length": episode_steps,
            }
        )

        epsilon = max(config["minimum_epsilon"], epsilon * config["epsilon_decay"])
        alpha = max(config["minimum_alpha"], alpha * config["alpha_decay"])

        if (episode_index + 1) % 250 == 0:
            recent_mean = np.mean([row["return"] for row in new_records[-100:]])
            print(
                f"Cumulative episode {episode_index + 1:5d}: "
                f"mean return (last 100)={recent_mean:8.2f}, "
                f"epsilon={epsilon:.3f}, alpha={alpha:.3f}"
            )

    env.close()
    new_history = pd.DataFrame(new_records)
    history = pd.concat([old_history, new_history], ignore_index=True)
    completed_episodes = end_episode

    if history["episode"].duplicated().any():
        raise ValueError("Duplicate episode numbers detected.")
    if int(history["episode"].iloc[-1]) != completed_episodes:
        raise ValueError("History does not end at the checkpoint episode count.")

    # Save only after the full training block succeeds.
    history.to_csv(HISTORY_PATH, index=False)
    save_latest_checkpoint(q_table, completed_episodes, epsilon, alpha, rng)
    return q_table, history, completed_episodes, epsilon, alpha


q_table, training_history, completed_episodes, epsilon, alpha = train_q_learning(
    CONFIG,
    resume_training=RESUME_TRAINING,
)

assert len(training_history) >= 1000, "Assignment minimum of 1,000 episodes was not met."
assert np.isfinite(q_table).all(), "Q-table contains NaN or infinite values."

print("Cumulative episodes completed:", completed_episodes)
print("Latest checkpoint:", CHECKPOINT_PATH.resolve())
print("Combined history:", HISTORY_PATH.resolve())
'''

notebook.cells[19].source += '''

# Preserve the strongest independently evaluated model across training blocks.
current_mean_return = float(trained_returns.mean())
if BEST_METADATA_PATH.exists():
    with open(BEST_METADATA_PATH, "r", encoding="utf-8") as file:
        best_metadata = json.load(file)
    previous_best_mean = float(best_metadata["mean_return"])
else:
    previous_best_mean = -np.inf

if current_mean_return > previous_best_mean:
    np.save(BEST_MODEL_PATH, q_table)
    best_metadata = {
        "mean_return": current_mean_return,
        "evaluation_episodes": int(CONFIG["evaluation_episodes"]),
        "completed_training_episodes": int(completed_episodes),
        "experiment_name": EXPERIMENT_NAME,
    }
    with open(BEST_METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(best_metadata, file, indent=2)
    print("New best model saved:", BEST_MODEL_PATH.resolve())
else:
    print(
        f"Latest mean {current_mean_return:.2f} did not exceed "
        f"best mean {previous_best_mean:.2f}; best model retained."
    )

best_q_table = np.load(BEST_MODEL_PATH)
'''

notebook.cells[22].source = notebook.cells[22].source.replace(
    "q_table[state]", "best_q_table[state]"
)

notebook.cells[23].source = '''## 12. Reproducibility and checkpoint checks

The checks below verify the latest checkpoint, cumulative history and best model. The latest checkpoint is used for future training; the best model is used for final evaluation/video when the latest block does not improve performance.
'''

notebook.cells[24].source = '''checkpoint = np.load(CHECKPOINT_PATH, allow_pickle=False)
reloaded_q_table = checkpoint["q_table"]
reloaded_completed_episodes = int(checkpoint["completed_episodes"])
reloaded_history = pd.read_csv(HISTORY_PATH)
reloaded_best_q_table = np.load(BEST_MODEL_PATH)

required_artifacts = [
    CHECKPOINT_PATH,
    HISTORY_PATH,
    BEST_MODEL_PATH,
    BEST_METADATA_PATH,
    RESULT_DIR / f"{EXPERIMENT_NAME}_config.json",
    RESULT_DIR / "pendulum_evaluation_summary.csv",
    RESULT_DIR / "pendulum_evaluation_episodes.csv",
    FIGURE_DIR / "pendulum_training_curves.png",
    FIGURE_DIR / "pendulum_policy_comparison.png",
]

assert np.array_equal(reloaded_q_table, q_table)
assert reloaded_q_table.shape == q_table_shape
assert reloaded_best_q_table.shape == q_table_shape
assert reloaded_completed_episodes == completed_episodes
assert len(reloaded_history) == completed_episodes
assert len(reloaded_history) >= 1000
assert reloaded_history["episode"].is_unique
assert all(path.exists() for path in required_artifacts)

print("All checkpoint and artifact checks passed.")
print("Cumulative episodes:", completed_episodes)
for path in required_artifacts:
    print(" -", path.resolve())
'''

notebook.cells[25].source = notebook.cells[25].source.replace(
    "The Q-Learning agent was trained for **3,000 episodes**",
    "Report the **cumulative episode count printed above**",
)

notebook.cells[26].source += '''

### How to continue training

1. Keep `EXPERIMENT_NAME` unchanged.
2. Set `RESUME_TRAINING = True`.
3. Set `TARGET_TOTAL_EPISODES` to the desired cumulative total, for example `80000`.
4. Run all cells. The code trains only the missing difference.
5. Confirm the printed cumulative count and checkpoint checks.

To start a genuinely new experiment, set `RESUME_TRAINING = False` **and change `EXPERIMENT_NAME`**. Do not reuse a checkpoint after changing bins, rewards or core hyperparameters.
'''

for cell in notebook.cells:
    if cell.cell_type == "code":
        cell.execution_count = None
        cell.outputs = []

nbformat.validate(notebook)
nbformat.write(notebook, output_path)
print(f"Wrote {output_path}")
