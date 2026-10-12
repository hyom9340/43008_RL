"""Generate self-contained CUDA notebooks for Assignment 2.

The generated notebooks contain no training outputs until run on the CUDA host.
Run this builder with a Python environment containing nbformat.
"""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
DEST = ROOT / "submission_cuda"
DEST.mkdir(exist_ok=True)


def notebook(title, sections):
    cells = [nbf.v4.new_markdown_cell(title)]
    for heading, explanation, code in sections:
        cells.append(nbf.v4.new_markdown_cell(f"## {heading}\n\n{explanation}"))
        cells.append(nbf.v4.new_code_cell(code.strip()))
    result = nbf.v4.new_notebook(cells=cells)
    result.metadata["kernelspec"] = {
        "display_name": "Python 3 (CUDA)", "language": "python", "name": "python3"
    }
    result.metadata["language_info"] = {"name": "python"}
    nbf.validate(result)
    return result


COMMON = r'''
from pathlib import Path
import csv, json, random, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
import torch as th
import torch.nn as nn
from IPython.display import display, Video
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv

SEED = 2026
random.seed(SEED)
np.random.seed(SEED)
th.manual_seed(SEED)
if not th.cuda.is_available():
    raise RuntimeError("Select the NVIDIA CUDA Python kernel before running training.")
DEVICE = "cuda:0"
print("Python:", sys.executable)
print("PyTorch:", th.__version__, "CUDA:", th.version.cuda)
print("GPU:", th.cuda.get_device_name(0))
'''


CNN = r'''
class CustomCNN(BaseFeaturesExtractor):
    """4x84x84 Atari frames -> 512 features; SB3 adds action/value heads."""
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            example = th.zeros((1, *observation_space.shape), dtype=th.float32)
            flattened = self.cnn(example).shape[1]
        self.linear = nn.Sequential(nn.Linear(flattened, features_dim), nn.ReLU())

    def forward(self, observations):
        return self.linear(self.cnn(observations.float() / 255.0))

CNN_POLICY = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=512),
    normalize_images=False,
)
print(CustomCNN)
'''


ANN = r'''
class CustomANN(BaseFeaturesExtractor):
    """CartPole's four state values -> 128 features; SB3 adds output heads."""
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        inputs = int(np.prod(observation_space.shape))
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(inputs, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, features_dim), nn.ReLU(),
        )

    def forward(self, observations):
        return self.net(observations.float())

ANN_POLICY = dict(
    features_extractor_class=CustomANN,
    features_extractor_kwargs=dict(features_dim=128),
)
print(CustomANN)
'''


BREAKOUT_ENV = r'''
import ale_py
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
gym.register_envs(ale_py)

def make_breakout(training=False, log_name=None, render_mode=None, append_log=False):
    def factory():
        env = gym.make("ALE/Breakout-v5", frameskip=1,
                       repeat_action_probability=0.0, render_mode=render_mode)
        env = AtariWrapper(env, screen_size=84, frame_skip=4,
                           terminal_on_life_loss=False, clip_reward=training)
        return Monitor(env, filename=str(OUT / "logs" / log_name) if log_name else None,
                       override_existing=not append_log)
    stacked = VecFrameStack(DummyVecEnv([factory]), n_stack=4, channels_order="last")
    return VecTransposeImage(stacked)

probe = make_breakout()
assert probe.observation_space.shape == (4, 84, 84)
print("Observation:", probe.observation_space, "Actions:", probe.action_space)
probe.close()
'''


BREAKOUT_EVAL_HELPERS = r'''
def evaluate_breakout(model, episodes=30, seed_offset=10_000):
    """Raw environment reward, deterministic policy, fixed episode seeds."""
    env = make_breakout(training=False)
    rows = []
    try:
        for i in range(episodes):
            seed = SEED + seed_offset + i
            env.seed(seed)
            env.action_space.seed(seed)
            obs = env.reset()
            total, length = 0.0, 0
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                total += float(reward[0])
                length += 1
                if done[0]:
                    break
            rows.append(dict(seed=seed, reward=total, length=length))
    finally:
        env.close()
    return pd.DataFrame(rows)

class FixedSeedEvaluation(BaseCallback):
    """Periodic reproducible evaluation and best-checkpoint selection."""
    def __init__(self, name, every=200_000, episodes=10):
        super().__init__()
        self.name, self.every, self.episodes = name, every, episodes
        self.history_path = OUT / "logs" / f"{name}_evaluation.csv"
        self.best_path = OUT / "models" / f"{name}_best"
        self.best_mean = -float("inf")
        if self.history_path.exists():
            old = pd.read_csv(self.history_path)
            eligible = old.loc[old.timesteps >= 1_000_000]
            if len(eligible):
                self.best_mean = float(eligible.mean_return.max())

    def _on_training_start(self):
        self.next_eval = (self.model.num_timesteps // self.every + 1) * self.every

    def _on_step(self):
        if self.model.num_timesteps < self.next_eval:
            return True
        data = evaluate_breakout(self.model, self.episodes)
        mean = float(data.reward.mean())
        row = dict(timesteps=self.model.num_timesteps, mean_return=mean,
                   std_return=float(data.reward.std(ddof=1)), episodes=self.episodes)
        old = pd.read_csv(self.history_path) if self.history_path.exists() else pd.DataFrame()
        pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(self.history_path, index=False)
        # A selected policy must itself have completed the Atari minimum.
        if self.model.num_timesteps >= 1_000_000 and mean > self.best_mean:
            self.best_mean = mean
            self.model.save(self.best_path)
        self.next_eval += self.every
        print(self.name, "evaluation:", row, "best:", round(self.best_mean, 2), flush=True)
        return True

class SaveCheckpoint(BaseCallback):
    def __init__(self, name, every=200_000, replay=False):
        super().__init__()
        self.name, self.every, self.replay = name, every, replay

    def _on_training_start(self):
        self.next_save = (self.model.num_timesteps // self.every + 1) * self.every

    def _on_step(self):
        if self.model.num_timesteps >= self.next_save:
            self.model.save(OUT / "models" / f"{self.name}_latest")
            if self.replay:
                self.model.save_replay_buffer(OUT / "models" / f"{self.name}_replay.pkl")
            self.next_save += self.every
        return True
'''


BREAKOUT_RANDOM = r'''
env = make_breakout(training=False)
random_rows = []
for i in range(30):
    seed = SEED + 10_000 + i
    env.seed(seed)
    env.action_space.seed(seed)
    obs = env.reset()
    total, length = 0.0, 0
    while True:
        obs, reward, done, _ = env.step(np.array([env.action_space.sample()]))
        total += float(reward[0]); length += 1
        if done[0]: break
    random_rows.append(dict(seed=seed, reward=total, length=length))
env.close()
random_results = pd.DataFrame(random_rows)
random_results.to_csv(OUT / "results" / "random_episodes.csv", index=False)
display(random_results.agg({"reward": ["mean", "std", "min", "max"], "length": ["mean", "std", "min", "max"]}))
'''


BREAKOUT_DQN = r'''
DQN_TARGET_STEPS = 3_600_000
dqn_path = OUT / "models" / "breakout_dqn_latest"
dqn_env = make_breakout(training=True, log_name="breakout_dqn_new",
                        append_log=dqn_path.with_suffix(".zip").exists())
if dqn_path.with_suffix(".zip").exists():
    dqn = DQN.load(dqn_path, env=dqn_env, device=DEVICE)
    replay_path = OUT / "models" / "breakout_dqn_replay.pkl"
    if dqn.num_timesteps < DQN_TARGET_STEPS:
        if not replay_path.exists(): raise FileNotFoundError("DQN resume needs the replay buffer")
        dqn.load_replay_buffer(replay_path)
else:
    dqn = DQN("CnnPolicy", dqn_env, policy_kwargs=CNN_POLICY, device=DEVICE,
              learning_rate=1e-4, buffer_size=50_000, learning_starts=50_000,
              batch_size=32, train_freq=4, target_update_interval=10_000,
              exploration_fraction=0.1, exploration_final_eps=0.01,
              optimize_memory_usage=True, replay_buffer_kwargs=dict(handle_timeout_termination=False),
              verbose=1, seed=SEED)
print("DQN CNN:", dqn.policy.features_extractor)
print("DQN parameters:", sum(p.numel() for p in dqn.policy.parameters()))
remaining = max(0, DQN_TARGET_STEPS - dqn.num_timesteps)
if remaining:
    dqn.learn(total_timesteps=remaining, reset_num_timesteps=False,
              callback=[FixedSeedEvaluation("breakout_dqn"), SaveCheckpoint("breakout_dqn", replay=True)])
    dqn.save(dqn_path)
    dqn.save_replay_buffer(OUT / "models" / "breakout_dqn_replay.pkl")
print("DQN actual steps:", dqn.num_timesteps)
dqn_env.close()
'''


BREAKOUT_PPO = r'''
# Controlled PPO revision: larger rollout (2048 vs 128), minibatches, learning-rate decay,
# periodic fixed-seed evaluation and best-checkpoint selection. Not a guaranteed improvement.
PPO_TARGET_STEPS = 6_000_000
ppo_path = OUT / "models" / "breakout_ppo_latest"
ppo_env = make_breakout(training=True, log_name="breakout_ppo_new",
                        append_log=ppo_path.with_suffix(".zip").exists())
if ppo_path.with_suffix(".zip").exists():
    ppo = PPO.load(ppo_path, env=ppo_env, device=DEVICE)
    if ppo.n_steps != 2048:
        raise ValueError("Existing PPO checkpoint uses a different rollout; use a fresh output directory")
else:
    ppo = PPO("CnnPolicy", ppo_env, policy_kwargs=CNN_POLICY, device=DEVICE,
              learning_rate=lambda progress: 2.5e-5 + 2.25e-4 * progress,
              n_steps=2048, batch_size=256, n_epochs=4, gamma=0.99,
              gae_lambda=0.95, clip_range=0.1, ent_coef=0.01,
              verbose=1, seed=SEED)
print("PPO CNN:", ppo.policy.features_extractor)
print("PPO parameters:", sum(p.numel() for p in ppo.policy.parameters()))
remaining = max(0, PPO_TARGET_STEPS - ppo.num_timesteps)
if remaining:
    ppo.learn(total_timesteps=remaining, reset_num_timesteps=False,
              callback=[FixedSeedEvaluation("breakout_ppo"), SaveCheckpoint("breakout_ppo")])
    ppo.save(ppo_path)
print("PPO actual steps:", ppo.num_timesteps)
ppo_env.close()
'''


BREAKOUT_PLOTS = r'''
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, name in zip(axes, ("breakout_dqn", "breakout_ppo")):
    path = OUT / "logs" / f"{name}_new.monitor.csv"
    training = pd.read_csv(path, comment="#")
    training["episode"] = np.arange(1, len(training) + 1)
    training["rolling100"] = training.r.rolling(100, min_periods=20).mean()
    ax.plot(training.episode, training.r, alpha=0.14, label="Training reward (clipped)")
    ax.plot(training.episode, training.rolling100, label="100-episode mean")
    ax.set(title=f"{name}: {len(training)} logged episodes", xlabel="Episode", ylabel="Clipped reward")
    ax.legend(); ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(OUT / "figures" / "breakout_training.png", dpi=180)
plt.show()

fig, ax = plt.subplots(figsize=(9, 5))
for name in ("breakout_dqn", "breakout_ppo"):
    data = pd.read_csv(OUT / "logs" / f"{name}_evaluation.csv")
    ax.errorbar(data.timesteps, data.mean_return, yerr=data.std_return,
                marker="o", capsize=2, label=name)
ax.axhline(random_results.reward.mean(), linestyle="--", color="gray", label="Random baseline")
ax.set(title="Fixed-seed evaluation during training", xlabel="Training time steps", ylabel="Raw episode reward")
ax.legend(); ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(OUT / "figures" / "breakout_evaluation_curve.png", dpi=180)
plt.show()
'''


BREAKOUT_FINAL = r'''
from stable_baselines3 import DQN, PPO
final_rows = []
for name, cls in (("breakout_dqn", DQN), ("breakout_ppo", PPO)):
    best = OUT / "models" / f"{name}_best.zip"
    if not best.exists(): raise FileNotFoundError(best)
    model = cls.load(best, device=DEVICE)
    episodes = evaluate_breakout(model, episodes=30)
    episodes.to_csv(OUT / "results" / f"{name}_episodes.csv", index=False)
    final_rows.append(dict(algorithm=name, model="best", train_timesteps=model.num_timesteps,
                           evaluation_episodes=len(episodes), mean_return=episodes.reward.mean(),
                           std_return=episodes.reward.std(ddof=1), median_return=episodes.reward.median()))
final_results = pd.DataFrame(final_rows)
final_results.to_csv(OUT / "results" / "breakout_summary.csv", index=False)
display(final_results)
print("Random mean:", random_results.reward.mean())

fig, ax = plt.subplots(figsize=(8, 5))
names = ["Random", "DQN", "PPO"]
groups = [random_results.reward,
          pd.read_csv(OUT / "results" / "breakout_dqn_episodes.csv").reward,
          pd.read_csv(OUT / "results" / "breakout_ppo_episodes.csv").reward]
ax.boxplot(groups, tick_labels=names, showmeans=True)
ax.set(title="Breakout: 30 fixed-seed evaluation episodes", ylabel="Raw episode reward")
ax.grid(axis="y", alpha=0.2)
fig.tight_layout()
fig.savefig(OUT / "figures" / "breakout_final_comparison.png", dpi=180)
plt.show()
'''


BREAKOUT_VIDEO = r'''
import imageio.v2 as imageio
for name, cls in (("breakout_dqn", DQN), ("breakout_ppo", PPO)):
    model = cls.load(OUT / "models" / f"{name}_best.zip", device=DEVICE)
    env = make_breakout(training=False, render_mode="rgb_array")
    env.seed(SEED + 40_000); obs = env.reset()
    path = OUT / "videos" / f"{name}_best.mp4"
    total, length = 0.0, 0
    try:
        with imageio.get_writer(path, fps=30, codec="libx264") as writer:
            while True:
                writer.append_data(np.asarray(env.render()))
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                total += float(reward[0]); length += 1
                if done[0]: break
    finally:
        env.close()
    print(name, "model=best", "seed=", SEED + 40_000,
          "episode reward=", total, "length=", length, "video=", path)
    display(Video(filename=str(path), embed=False))
'''


BREAKOUT_CHECK = r'''
required = [OUT / "results" / "breakout_summary.csv",
            OUT / "figures" / "breakout_training.png",
            OUT / "figures" / "breakout_evaluation_curve.png",
            OUT / "figures" / "breakout_final_comparison.png"]
for name in ("breakout_dqn", "breakout_ppo"):
    required += [OUT / "models" / f"{name}_best.zip",
                 OUT / "videos" / f"{name}_best.mp4",
                 OUT / "results" / f"{name}_episodes.csv"]
missing = [str(p) for p in required if not p.exists()]
assert not missing, f"Missing artifacts: {missing}"
assert all(final_results.train_timesteps >= 1_000_000)
print("All Breakout artifacts exist. Evaluation table above is the report source.")
'''


breakout_sections = [
    ("1. CUDA setup and reproducibility", "Choose the NVIDIA CUDA kernel. Outputs use a fresh folder and do not mix historical checkpoints.",
     COMMON + '\nOUT = Path.cwd() / "submission_cuda_outputs" / "breakout"\nfor part in ("models", "logs", "results", "figures", "videos"):\n    (OUT / part).mkdir(parents=True, exist_ok=True)\nprint("Output:", OUT)'),
    ("2. Environment", "Breakout uses four 84×84 grayscale frames. Training reward is clipped; evaluation uses raw reward.", BREAKOUT_ENV),
    ("3. Random baseline", "Thirty episodes use the same seeds as the final trained-policy evaluation.", BREAKOUT_RANDOM),
    ("4. Custom CNN", "The CNN is the student's custom SB3 feature extractor.", CNN),
    ("5. Evaluation and checkpoint helpers", "Every 200,000 training steps, test on ten fixed seeds and retain the best checkpoint.", BREAKOUT_EVAL_HELPERS),
    ("6. Train DQN", "Independent DQN run. The replay buffer is saved for resumable training.", BREAKOUT_DQN),
    ("7. Train PPO — revised configuration", "Hypothesis: a 2,048-step rollout and minibatch updates may improve stability versus the prior 128-step rollout. This requires empirical validation.", BREAKOUT_PPO),
    ("8. Training and evaluation curves", "Never splice the old interleaved Monitor CSV into these new curves.", BREAKOUT_PLOTS),
    ("9. Independent final evaluation", "Reload each best saved checkpoint and test on 30 fixed seeds.", BREAKOUT_FINAL),
    ("10. MP4 simulations", "One complete deterministic episode per best checkpoint; video score is not the 30-episode mean.", BREAKOUT_VIDEO),
    ("11. Submission checks", "Check the evidence files and minimum Atari training steps.", BREAKOUT_CHECK),
]


CARTPOLE_ENV = r'''
def make_cartpole(log_name=None, render_mode=None):
    def factory():
        env = gym.make("CartPole-v1", render_mode=render_mode)
        return Monitor(env, filename=str(OUT / "logs" / log_name) if log_name else None)
    return DummyVecEnv([factory])

probe = gym.make("CartPole-v1")
print("State:", probe.observation_space, "Actions:", probe.action_space)
probe.close()
'''


CARTPOLE_EVAL = r'''
def evaluate_cartpole(model, episodes=100, random_policy=False, seed_offset=10_000):
    env = gym.make("CartPole-v1")
    rows = []
    try:
        for i in range(episodes):
            seed = SEED + seed_offset + i
            obs, _ = env.reset(seed=seed)
            env.action_space.seed(seed)
            total, length = 0.0, 0
            while True:
                action = env.action_space.sample() if random_policy else int(model.predict(obs, deterministic=True)[0])
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward); length += 1
                if terminated or truncated: break
            rows.append(dict(seed=seed, reward=total, length=length))
    finally:
        env.close()
    return pd.DataFrame(rows)

class EpisodeTarget(BaseCallback):
    def __init__(self, target):
        super().__init__(); self.target = target; self.episodes = 0
    def _on_step(self):
        self.episodes += int(np.asarray(self.locals["dones"]).sum())
        return self.episodes < self.target
'''


CARTPOLE_DQN = r'''
CARTPOLE_TARGET_EPISODES = 10_000
dqn_env = make_cartpole(log_name="cartpole_dqn")
dqn = DQN("MlpPolicy", dqn_env, policy_kwargs=ANN_POLICY, device=DEVICE,
          learning_rate=1e-3, buffer_size=100_000, learning_starts=1_000,
          batch_size=64, gamma=0.99, exploration_fraction=0.2,
          exploration_final_eps=0.05, verbose=1, seed=SEED)
print("DQN ANN:", dqn.policy.features_extractor)
print("DQN parameters:", sum(p.numel() for p in dqn.policy.parameters()))
dqn_counter = EpisodeTarget(CARTPOLE_TARGET_EPISODES)
dqn.learn(total_timesteps=10_000_000, callback=dqn_counter)
dqn.save(OUT / "models" / "cartpole_dqn_final")
dqn_training = dict(algorithm="cartpole_dqn", episodes=dqn_counter.episodes,
                    timesteps=dqn.num_timesteps)
dqn_env.close()
display(pd.DataFrame([dqn_training]))
'''


CARTPOLE_PPO = r'''
ppo_env = make_cartpole(log_name="cartpole_ppo")
ppo = PPO("MlpPolicy", ppo_env, policy_kwargs=ANN_POLICY, device=DEVICE,
          learning_rate=3e-4, n_steps=2_048, batch_size=64,
          gamma=0.99, verbose=1, seed=SEED)
print("PPO ANN:", ppo.policy.features_extractor)
print("PPO parameters:", sum(p.numel() for p in ppo.policy.parameters()))
ppo_counter = EpisodeTarget(CARTPOLE_TARGET_EPISODES)
ppo.learn(total_timesteps=10_000_000, callback=ppo_counter)
ppo.save(OUT / "models" / "cartpole_ppo_final")
ppo_training = dict(algorithm="cartpole_ppo", episodes=ppo_counter.episodes,
                    timesteps=ppo.num_timesteps)
ppo_env.close()
training_summary = pd.DataFrame([dqn_training, ppo_training])
training_summary.to_csv(OUT / "results" / "cartpole_training_summary.csv", index=False)
display(training_summary)
assert all(training_summary.episodes >= 1_000)
'''


CARTPOLE_PLOTS = r'''
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, name in zip(axes, ("cartpole_dqn", "cartpole_ppo")):
    data = pd.read_csv(OUT / "logs" / f"{name}.monitor.csv", comment="#")
    data["episode"] = np.arange(1, len(data) + 1)
    data["rolling100"] = data.r.rolling(100, min_periods=20).mean()
    ax.plot(data.episode, data.r, alpha=0.12, label="Episode reward")
    ax.plot(data.episode, data.rolling100, label="100-episode mean")
    ax.set(title=f"{name}: {len(data)} episodes", xlabel="Episode", ylabel="Reward")
    ax.grid(alpha=0.2); ax.legend()
fig.tight_layout()
fig.savefig(OUT / "figures" / "cartpole_training.png", dpi=180)
plt.show()
'''


CARTPOLE_FINAL = r'''
random_results = evaluate_cartpole(None, random_policy=True)
random_results.to_csv(OUT / "results" / "random_episodes.csv", index=False)
rows = [dict(algorithm="random", train_episodes=0, evaluation_episodes=100,
             mean_return=random_results.reward.mean(), std_return=random_results.reward.std(ddof=1))]
groups = [random_results.reward]
for name, cls in (("cartpole_dqn", DQN), ("cartpole_ppo", PPO)):
    model = cls.load(OUT / "models" / f"{name}_final.zip", device=DEVICE)
    result = evaluate_cartpole(model)
    result.to_csv(OUT / "results" / f"{name}_episodes.csv", index=False)
    train_episodes = int(training_summary.loc[training_summary.algorithm == name, "episodes"].iloc[0])
    rows.append(dict(algorithm=name, train_episodes=train_episodes, evaluation_episodes=len(result),
                     mean_return=result.reward.mean(), std_return=result.reward.std(ddof=1)))
    groups.append(result.reward)
final_results = pd.DataFrame(rows)
final_results.to_csv(OUT / "results" / "cartpole_summary.csv", index=False)
display(final_results)
fig, ax = plt.subplots(figsize=(8, 5))
ax.boxplot(groups, tick_labels=["Random", "DQN", "PPO"], showmeans=True)
ax.set(title="CartPole: 100 fixed-seed evaluation episodes", ylabel="Episode reward")
ax.grid(axis="y", alpha=0.2)
fig.tight_layout()
fig.savefig(OUT / "figures" / "cartpole_comparison.png", dpi=180)
plt.show()
'''


CARTPOLE_VIDEO = r'''
from gymnasium.wrappers import RecordVideo
for name, cls in (("cartpole_dqn", DQN), ("cartpole_ppo", PPO)):
    model = cls.load(OUT / "models" / f"{name}_final.zip", device=DEVICE)
    env = RecordVideo(gym.make("CartPole-v1", render_mode="rgb_array"),
                      video_folder=str(OUT / "videos"), name_prefix=name,
                      episode_trigger=lambda episode: episode == 0, disable_logger=True)
    obs, _ = env.reset(seed=SEED + 40_000)
    total, length = 0.0, 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        total += float(reward); length += 1
        if terminated or truncated: break
    env.close()
    path = OUT / "videos" / f"{name}-episode-0.mp4"
    print(name, "reward=", total, "length=", length, "seed=", SEED + 40_000, path)
    display(Video(filename=str(path), embed=False))
'''


CARTPOLE_CHECK = r'''
required = [OUT / "results" / "cartpole_summary.csv",
            OUT / "figures" / "cartpole_training.png",
            OUT / "figures" / "cartpole_comparison.png"]
for name in ("cartpole_dqn", "cartpole_ppo"):
    required += [OUT / "models" / f"{name}_final.zip",
                 OUT / "videos" / f"{name}-episode-0.mp4",
                 OUT / "results" / f"{name}_episodes.csv"]
missing = [str(p) for p in required if not p.exists()]
assert not missing, f"Missing artifacts: {missing}"
assert all(training_summary.episodes >= 1_000)
print("All CartPole artifacts exist. Evaluation table above is the report source.")
'''


cartpole_sections = [
    ("1. CUDA setup and reproducibility", "Choose the NVIDIA CUDA kernel. This run writes only to the new output folder.",
     COMMON + '\nOUT = Path.cwd() / "submission_cuda_outputs" / "cartpole"\nfor part in ("models", "logs", "results", "figures", "videos"):\n    (OUT / part).mkdir(parents=True, exist_ok=True)\nprint("Output:", OUT)'),
    ("2. Environment", "The observation is four numeric state values; the action space is discrete.", CARTPOLE_ENV),
    ("3. Evaluation and episode counter", "Use the same 100 seeds for random, DQN and PPO.", CARTPOLE_EVAL),
    ("4. Custom ANN", "Both algorithms use this custom SB3 feature extractor.", ANN),
    ("5. Train DQN", "Run DQN to 10,000 completed episodes.", CARTPOLE_DQN),
    ("6. Train PPO", "Run PPO independently to 10,000 completed episodes.", CARTPOLE_PPO),
    ("7. Training curves", "Training reward and a 100-episode moving average are shown separately.", CARTPOLE_PLOTS),
    ("8. Independent final evaluation", "Reload saved models and compare against the random baseline.", CARTPOLE_FINAL),
    ("9. MP4 simulations", "One complete deterministic episode for each trained algorithm.", CARTPOLE_VIDEO),
    ("10. Submission checks", "Verify the models, results, figures and videos.", CARTPOLE_CHECK),
]


nbf.write(notebook("# Assignment 2 — Breakout (CUDA, DQN and PPO)\n\nRun top to bottom; save this notebook **with outputs** before submission.", breakout_sections), DEST / "Breakout_DQN_PPO_CUDA.ipynb")
nbf.write(notebook("# Assignment 2 — CartPole (CUDA, DQN and PPO)\n\nRun top to bottom; save this notebook **with outputs** before submission.", cartpole_sections), DEST / "CartPole_DQN_PPO_CUDA.ipynb")

# The validated Pendulum experiment is kept intact, including its actual outputs.
pendulum = nbf.read(ROOT / "Pendulum_TD_Q_Learning.ipynb", as_version=4)
nbf.validate(pendulum)
nbf.write(pendulum, DEST / "Pendulum_Q_Learning.ipynb")

print("Generated three notebooks in", DEST)
