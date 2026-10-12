from pathlib import Path
import json

import numpy as np
import pandas as pd


root = Path(__file__).resolve().parent
model_dir = root / "pendulum_artifacts" / "models"
result_dir = root / "pendulum_artifacts" / "results"

experiment_name = "pendulum_qlearning_36x36x15_seed2026_from50000"
legacy_q_path = model_dir / "pendulum_q_table.npy"
legacy_history_path = result_dir / "pendulum_training_history.csv"
checkpoint_path = model_dir / f"{experiment_name}_latest_checkpoint.npz"
history_path = result_dir / f"{experiment_name}_training_history.csv"

q_table = np.load(legacy_q_path)
history = pd.read_csv(legacy_history_path)

if q_table.shape != (36, 36, 15):
    raise ValueError(f"Unexpected legacy Q-table shape: {q_table.shape}")
if len(history) != 50_000:
    raise ValueError(f"Expected 50,000 history rows, found {len(history)}")
if int(history["episode"].iloc[-1]) != 50_000:
    raise ValueError("Legacy history does not end at episode 50,000")
if history["episode"].duplicated().any():
    raise ValueError("Legacy history contains duplicate episode numbers")

# At 50,000 episodes both schedules have reached their configured minima.
epsilon = max(0.03, 1.0 * (0.9975 ** 50_000))
alpha = max(0.03, 0.18 * (0.999 ** 50_000))

# The old notebook did not save its RNG state. Start a deterministic continuation
# stream and save it from this point onward. This does not alter the learned Q-table.
rng = np.random.default_rng(2026 + 50_000)

np.savez_compressed(
    checkpoint_path,
    q_table=q_table.astype(np.float32, copy=False),
    completed_episodes=np.int64(50_000),
    epsilon=np.float64(epsilon),
    alpha=np.float64(alpha),
    rng_state=np.array(json.dumps(rng.bit_generator.state)),
)
history.to_csv(history_path, index=False)

print("Migrated episodes:", len(history))
print("Checkpoint:", checkpoint_path)
print("History:", history_path)
print("epsilon:", epsilon, "alpha:", alpha)
