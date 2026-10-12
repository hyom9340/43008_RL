from copy import deepcopy
from pathlib import Path
import nbformat

root = Path(__file__).resolve().parent
source = nbformat.read(root / "Assignment2_Remaining_GPU_AMD_DirectML.ipynb", as_version=4)

def save(name, indices, title, artifact_subdir, extra_cells=None):
    nb = nbformat.v4.new_notebook()
    nb.metadata = deepcopy(source.metadata)
    nb.cells = [deepcopy(source.cells[i]) for i in indices]
    nb.cells[0].source = title
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.source = cell.source.replace(
                'OUT = ROOT / "remaining_artifacts" / "gpu_amd_directml"',
                f'OUT = ROOT / "remaining_artifacts" / "gpu_amd_directml" / "{artifact_subdir}"',
            )
            cell.execution_count = None
            cell.outputs = []
    if extra_cells:
        nb.cells.extend(extra_cells)
    nbformat.validate(nb)
    nbformat.write(nb, root / name)

cart_title = '''# 43008 Assignment 2 - CartPole DQN and PPO (AMD DirectML GPU)

**Student:** replace name and ID  
**Environment:** Gymnasium `CartPole-v1`  
**Algorithms:** DQN and PPO  
**Custom extractor:** student-defined ANN

This notebook has intentionally not been executed. Run it top-to-bottom and retain all outputs. Each algorithm counts actual completed episodes and targets at least 1,000 episodes.'''

save(
    "Assignment2_CartPole_DQN_PPO_GPU_AMD_DirectML.ipynb",
    list(range(0, 11)) + [15, 16, 17],
    cart_title,
    "cartpole",
)

breakout_title = '''# 43008 Assignment 2 - Breakout DQN and PPO (AMD DirectML GPU)

**Student:** replace name and ID  
**Environment:** Gymnasium/ALE `Breakout`  
**Algorithms:** DQN and PPO  
**Custom extractor:** student-defined CNN

This notebook has intentionally not been executed. Run it top-to-bottom and retain all outputs. Each algorithm targets at least 1,000,000 cumulative timesteps and must be evaluated against a random baseline.'''

random_baseline = nbformat.v4.new_code_cell('''def breakout_random_baseline(episodes=30):
    env = make_breakout(False)
    returns = []
    for episode in range(episodes):
        obs = env.reset(); done = np.array([False]); total = 0.0
        while not done[0]:
            action = np.array([env.action_space.sample()])
            obs, reward, done, info = env.step(action)
            total += float(reward[0])
        returns.append(total)
    env.close()
    returns = np.asarray(returns)
    print(f"Random Breakout mean={returns.mean():.2f}, std={returns.std(ddof=1):.2f}")
    return returns

breakout_random_returns = breakout_random_baseline(30)''')

breakout_outputs = nbformat.v4.new_code_cell('''def record_breakout(model_path, algorithm_class, name, max_steps=10000):
    env = make_breakout(False)
    model = algorithm_class.load(model_path, env=env, device=DEVICE)
    video = VecVideoRecorder(
        env, str(OUT/"videos"), record_video_trigger=lambda step: step == 0,
        video_length=max_steps, name_prefix=name,
    )
    obs = video.reset()
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = video.step(action)
        if done[0]: break
    video.close()

# Run after both training cells have completed:
# record_breakout(OUT/"models"/"breakout_dqn_latest", DQN, "breakout-dqn")
# record_breakout(OUT/"models"/"breakout_ppo_latest", PPO, "breakout-ppo")

pd.DataFrame(atari_results).to_csv(OUT/"breakout_evaluation.csv", index=False)
print("Evaluation table saved to", OUT/"breakout_evaluation.csv")''')

# Shared setup/extractors/callbacks, then Breakout-specific environment and training.
breakout_indices = list(range(0, 9)) + [11, 12]
nb = nbformat.v4.new_notebook()
nb.metadata = deepcopy(source.metadata)
nb.cells = [deepcopy(source.cells[i]) for i in breakout_indices]
nb.cells[0].source = breakout_title
nb.cells.append(random_baseline)
nb.cells.extend([deepcopy(source.cells[13]), deepcopy(source.cells[14]), deepcopy(source.cells[15]), breakout_outputs, deepcopy(source.cells[17])])
for cell in nb.cells:
    if cell.cell_type == "code":
        cell.source = cell.source.replace(
            'OUT = ROOT / "remaining_artifacts" / "gpu_amd_directml"',
            'OUT = ROOT / "remaining_artifacts" / "gpu_amd_directml" / "breakout"',
        )
        cell.execution_count = None
        cell.outputs = []
nbformat.validate(nb)
nbformat.write(nb, root / "Assignment2_Breakout_DQN_PPO_GPU_AMD_DirectML.ipynb")

print("Split GPU notebooks written")
