from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent

def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)

def build(gpu=False):
    mode = "GPU_AMD_DirectML" if gpu else "CPU"
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    install = '''# Run once, then restart the kernel.
%pip install "gymnasium[classic-control,atari,accept-rom-license]" "stable-baselines3[extra]" ale-py autorom moviepy imageio-ffmpeg pandas matplotlib''' + (''' torch-directml''' if gpu else '')
    device = '''import torch_directml
DEVICE = torch_directml.device()
if "privateuseone" not in str(DEVICE):
    raise RuntimeError("DirectML device was not created; do not continue on CPU silently.")
print("DirectML device:", DEVICE)''' if gpu else '''DEVICE = "cpu"
print("Training device:", DEVICE)'''
    cells = [
        md(f'''# 43008 Assignment 2 - Remaining Experiments ({mode})

**Student:** replace name and ID  
**Experiments:** CartPole DQN, CartPole PPO, Breakout DQN, Breakout PPO  
**Custom extractors:** custom ANN for CartPole and custom CNN for Breakout.

This notebook contains training code only and has intentionally not been executed. Run it top-to-bottom and retain all outputs before submission. CartPole targets at least 1,000 completed episodes per algorithm. Breakout targets at least 1,000,000 timesteps per algorithm.'''),
        md('''## 1. Setup and reproducibility

CartPole has a four-value continuous observation and two discrete actions. Breakout uses stacked image frames and a discrete action set. DQN learns action values with replay and a target network; PPO directly optimises a clipped actor-critic objective.'''),
        code(install),
        code('''from pathlib import Path
import json, os, random, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
import ale_py
import torch as th
import torch.nn as nn
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecVideoRecorder
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.evaluation import evaluate_policy

gym.register_envs(ale_py)
SEED = 2026
random.seed(SEED); np.random.seed(SEED); th.manual_seed(SEED)
ROOT = Path.cwd()
OUT = ROOT / "remaining_artifacts" / "''' + mode.lower() + '''"
for p in [OUT/"models", OUT/"logs", OUT/"figures", OUT/"videos"]: p.mkdir(parents=True, exist_ok=True)
print("Python/PyTorch/Gymnasium:", th.__version__, gym.__version__)'''),
        code(device),
        md('''## 2. Custom feature extractors

These are student-defined extractors passed through `policy_kwargs`, satisfying the custom ANN/CNN requirement. Network diagrams: `4 -> 128 -> 128 -> 128 features` for CartPole; `stacked frames -> Conv(32) -> Conv(64) -> Conv(64) -> 512 features` for Breakout.'''),
        code('''class CustomANN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        n = int(np.prod(observation_space.shape))
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(n,128), nn.ReLU(), nn.Linear(128,128), nn.ReLU(), nn.Linear(128,features_dim), nn.ReLU())
    def forward(self, observations): return self.net(observations.float())

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        c = observation_space.shape[0]
        self.cnn = nn.Sequential(nn.Conv2d(c,32,8,4),nn.ReLU(),nn.Conv2d(32,64,4,2),nn.ReLU(),nn.Conv2d(64,64,3,1),nn.ReLU(),nn.Flatten())
        with th.no_grad(): flat = self.cnn(th.as_tensor(observation_space.sample()[None]).float()/255.0).shape[1]
        self.linear = nn.Sequential(nn.Linear(flat,features_dim),nn.ReLU())
    def forward(self, observations): return self.linear(self.cnn(observations.float()/255.0))

ANN_KW = dict(features_extractor_class=CustomANN, features_extractor_kwargs=dict(features_dim=128))
CNN_KW = dict(features_extractor_class=CustomCNN, features_extractor_kwargs=dict(features_dim=512), normalize_images=False)'''),
        md('''## 3. Checkpoint callback

Models save periodically and resume toward a fixed target. DQN replay-buffer saving is enabled; it can be large but preserves genuine off-policy continuation.'''),
        code('''class SaveProgress(BaseCallback):
    def __init__(self, algorithm, folder, save_freq, save_replay=False):
        super().__init__(); self.algorithm=algorithm; self.folder=Path(folder); self.save_freq=save_freq; self.save_replay=save_replay; self.next_save=save_freq
    def _on_step(self):
        if self.num_timesteps >= self.next_save:
            self.model.save(self.folder/f"{self.algorithm}_latest")
            if self.save_replay and hasattr(self.model,"save_replay_buffer"): self.model.save_replay_buffer(self.folder/f"{self.algorithm}_replay.pkl")
            self.next_save += self.save_freq
        return True

def load_or_create(cls, path, env, kwargs):
    if Path(str(path)+".zip").exists():
        model=cls.load(path, env=env, device=DEVICE)
        replay=Path(str(path).replace("_latest", "_replay.pkl"))
        if cls is DQN and replay.exists(): model.load_replay_buffer(replay)
        print("Resumed", path, "at timestep", model.num_timesteps)
        return model
    return cls(env=env, device=DEVICE, verbose=1, seed=SEED, **kwargs)'''),
        md('''## 4. CartPole: random baseline, DQN and PPO

Each algorithm must complete at least 1,000 episodes. Because SB3 trains in timesteps, the callback counts actual completed episodes and stops only at the cumulative target.'''),
        code('''def random_baseline(env_id, episodes=100):
    env=gym.make(env_id); scores=[]
    for ep in range(episodes):
        obs,_=env.reset(seed=10000+ep); done=False; total=0
        while not done:
            obs,r,term,trunc,_=env.step(env.action_space.sample()); total+=r; done=term or trunc
        scores.append(total)
    env.close(); return np.array(scores)

class EpisodeTarget(BaseCallback):
    def __init__(self, target, state_path): super().__init__(); self.target=target; self.state_path=Path(state_path); self.count=0
    def _on_training_start(self):
        if self.state_path.exists(): self.count=json.loads(self.state_path.read_text())["episodes"]
    def _on_step(self):
        self.count += int(np.asarray(self.locals["dones"]).sum())
        if self.count % 25 == 0: self.state_path.write_text(json.dumps({"episodes":self.count}))
        return self.count < self.target
    def _on_training_end(self): self.state_path.write_text(json.dumps({"episodes":self.count}))

CARTPOLE_TARGET_EPISODES=1000
cart_random=random_baseline("CartPole-v1",100)
cart_results=[]
for name,cls,kwargs in [
    ("cartpole_dqn",DQN,dict(policy="MlpPolicy",policy_kwargs=ANN_KW,learning_rate=1e-3,buffer_size=100000,learning_starts=1000,batch_size=64,gamma=.99,exploration_fraction=.2,exploration_final_eps=.05)),
    ("cartpole_ppo",PPO,dict(policy="MlpPolicy",policy_kwargs=ANN_KW,learning_rate=3e-4,n_steps=2048,batch_size=64,gamma=.99)),
]:
    env=DummyVecEnv([lambda n=name: Monitor(gym.make("CartPole-v1"),str(OUT/"logs"/n))])
    latest=OUT/"models"/f"{name}_latest"; model=load_or_create(cls,latest,env,kwargs)
    stop=EpisodeTarget(CARTPOLE_TARGET_EPISODES,OUT/"logs"/f"{name}_episodes.json")
    saver=SaveProgress(name,OUT/"models",25000,save_replay=(cls is DQN))
    model.learn(total_timesteps=1_000_000,callback=[stop,saver],reset_num_timesteps=False)
    model.save(latest)
    if cls is DQN: model.save_replay_buffer(OUT/"models"/f"{name}_replay.pkl")
    mean,std=evaluate_policy(model,env,n_eval_episodes=100,deterministic=True)
    cart_results.append(dict(policy=name,mean_return=mean,std_return=std,episodes=stop.count,timesteps=model.num_timesteps))
    env.close()
display(pd.DataFrame(cart_results)); print("Random mean/std",cart_random.mean(),cart_random.std(ddof=1))'''),
        md('''## 5. Breakout environment and preprocessing

Training uses 84x84 grayscale observations, action repeat/max-pooling, frame stacking and reward clipping. Evaluation disables reward clipping so reported scores remain in the game's original reward scale.'''),
        code('''def make_breakout(training=True):
    def factory():
        env=gym.make("ALE/Breakout-v5",frameskip=1,repeat_action_probability=0.0)
        env=AtariWrapper(env,screen_size=84,frame_skip=4,terminal_on_life_loss=False,clip_reward=training)
        return Monitor(env)
    return VecFrameStack(DummyVecEnv([factory]),n_stack=4,channels_order="last")

test_env=make_breakout(False)
print("Breakout observation/action:",test_env.observation_space,test_env.action_space)
test_env.close()'''),
        md('''## 6. Breakout DQN and PPO training

Both targets are fixed at 1,000,000 cumulative timesteps. Re-running resumes only the missing difference. Increase the target if the agent does not beat the random baseline.'''),
        code('''ATARI_TARGET_TIMESTEPS=1_000_000
atari_results=[]
for name,cls,kwargs in [
    ("breakout_dqn",DQN,dict(policy="CnnPolicy",policy_kwargs=CNN_KW,learning_rate=1e-4,buffer_size=50000,learning_starts=50000,batch_size=32,gamma=.99,train_freq=4,target_update_interval=10000,exploration_fraction=.1,exploration_final_eps=.01,optimize_memory_usage=True)),
    ("breakout_ppo",PPO,dict(policy="CnnPolicy",policy_kwargs=CNN_KW,learning_rate=2.5e-4,n_steps=128,batch_size=256,n_epochs=4,gamma=.99,gae_lambda=.95,clip_range=.1,ent_coef=.01)),
]:
    env=make_breakout(True); latest=OUT/"models"/f"{name}_latest"; model=load_or_create(cls,latest,env,kwargs)
    remaining=max(0,ATARI_TARGET_TIMESTEPS-model.num_timesteps)
    if remaining: model.learn(total_timesteps=remaining,callback=SaveProgress(name,OUT/"models",100000,save_replay=(cls is DQN)),reset_num_timesteps=False)
    model.save(latest)
    if cls is DQN: model.save_replay_buffer(OUT/"models"/f"{name}_replay.pkl")
    eval_env=make_breakout(False); mean,std=evaluate_policy(model,eval_env,n_eval_episodes=30,deterministic=True)
    atari_results.append(dict(policy=name,mean_return=mean,std_return=std,timesteps=model.num_timesteps))
    env.close(); eval_env.close()
display(pd.DataFrame(atari_results))'''),
        md('''## 7. Curves, videos and submission outputs

Use Monitor CSV files to plot episode reward and a 100-episode rolling mean. Record one deterministic episode for each final model and report mean ± standard deviation against the random baseline. Retain model ZIPs, replay buffer, CSV logs, figures and MP4 files.'''),
        code('''def plot_monitor(csv_path,title,out_path):
    data=pd.read_csv(csv_path,comment="#"); data["rolling100"]=data["r"].rolling(100,min_periods=1).mean()
    ax=data.plot(y=["r","rolling100"],alpha=.7,title=title); ax.set(xlabel="Episode",ylabel="Return"); plt.savefig(out_path,dpi=180,bbox_inches="tight"); plt.show()

def record_cartpole(model_path,cls,name):
    env=DummyVecEnv([lambda: gym.make("CartPole-v1",render_mode="rgb_array")]); model=cls.load(model_path,env=env,device=DEVICE)
    video=VecVideoRecorder(env,str(OUT/"videos"),record_video_trigger=lambda x:x==0,video_length=500,name_prefix=name)
    obs=video.reset()
    for _ in range(500):
        action,_=model.predict(obs,deterministic=True); obs,_,done,_=video.step(action)
        if done[0]: break
    video.close()

# Run after training:
# plot_monitor(OUT/"logs"/"cartpole_dqn.monitor.csv","CartPole DQN",OUT/"figures"/"cartpole_dqn.png")
# record_cartpole(OUT/"models"/"cartpole_dqn_latest",DQN,"cartpole-dqn")'''),
        md('''## 8. Report discussion checklist

- State/action/reward/termination for both environments.
- DQN and PPO formulas, pseudocode, flowcharts and the custom network diagrams.
- Exact hyperparameters, device, seeds, episode/timestep totals and wall-clock time.
- Learning curves and independent evaluation mean, standard deviation and random baseline.
- Compare sample efficiency, stability, exploration, replay/off-policy versus on-policy learning.
- Confirm both Atari agents exceed random performance; train beyond one million timesteps if necessary.
- Keep notebook outputs and report numbers identical; include simulation MP4 and AI-use declaration.''')
    ]
    nb.cells=cells
    path=ROOT/f"Assignment2_Remaining_{mode}.ipynb"
    nbf.validate(nb); nbf.write(nb,path); print(path)

build(False); build(True)
