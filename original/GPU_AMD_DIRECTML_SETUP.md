# Local GPU setup for the assignment notebooks

## Detected hardware

- GPU: AMD Radeon(TM) Graphics
- Reported dedicated/shared adapter memory: approximately 2 GB
- NVIDIA CUDA: unavailable (`nvidia-smi` was not found)
- Windows driver: 32.0.21041.1000

The GPU notebook therefore uses Microsoft DirectML through `torch-directml`, not CUDA. DirectML support is more experimental than CUDA and may be slower than CPU for CartPole. Breakout's CNN is the experiment most likely to benefit, but 2 GB memory is restrictive.

## Create a separate environment

In the VS Code terminal, from `C:\Users\James\Desktop\RL`:

```powershell
py -3.12 -m venv .venv-directml
.\.venv-directml\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch-directml
python -m pip install "gymnasium[classic-control,atari,accept-rom-license]" "stable-baselines3[extra]" ale-py autorom moviepy imageio-ffmpeg pandas matplotlib ipykernel
python -m ipykernel install --user --name rl-directml --display-name "Python (RL DirectML)"
```

If PowerShell blocks activation, the environment can still be used directly:

```powershell
.\.venv-directml\Scripts\python.exe -m pip install torch-directml
```

## VS Code

1. Open `Assignment2_Remaining_GPU_AMD_DirectML.ipynb`.
2. Select the kernel `Python (RL DirectML)`.
3. Run only the setup/device cells first.
4. Confirm the printed device contains `privateuseone`. This is PyTorch's DirectML device name.
5. If it reports CPU or raises an error, do not run the long training cells.

## Memory precautions

- Close browsers and other GPU-heavy applications.
- Keep the Atari vector environment count at one.
- The supplied DQN replay buffer is 50,000 transitions; its saved file can still be large.
- If an out-of-memory error occurs, reduce `buffer_size` to 20,000 and PPO `batch_size` to 128. Record any change in the report.
- Do not increase CNN width or the number of parallel environments on this 2 GB GPU.

## Important limitations

- Stable-Baselines3 primarily targets CPU and CUDA. DirectML may encounter unsupported operations.
- CartPole's small ANN generally runs faster on CPU; the GPU version is provided for experimental parity.
- Do not claim GPU acceleration unless the notebook prints a DirectML device and training completes on it.
- If DirectML fails, use the CPU notebook locally or use an NVIDIA CUDA machine/AWS for Breakout.

## Verify without starting training

```python
import torch
import torch_directml

device = torch_directml.device()
x = torch.randn(256, 256, device=device)
y = x @ x
print(device, y.device, y.shape)
```

This test checks that PyTorch can execute a matrix operation on DirectML. It does not run the reinforcement-learning experiments.
