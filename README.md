# Reinforcement Learning Assignment 2

Local code and notebooks for the Atari, Classic Control, and Pendulum experiments.

## Run locally

Open this folder in VS Code, select the project's `.venv` Python interpreter, and run the notebooks from top to bottom. Install dependencies from `requirements.txt`; the AMD DirectML setup is described in `GPU_AMD_DIRECTML_SETUP.md`.

Training outputs, model checkpoints, videos, local virtual environments, and submission ZIP files are excluded from Git. Keep those files locally or in a separate backup. The executed `.ipynb` files retain their cell outputs when saved.

## NVIDIA CUDA submission notebooks

The revised three-notebook workflow is in [submission_cuda/README.md](submission_cuda/README.md). Run `build_cuda_submission.py` to regenerate the notebooks. Breakout PPO uses a new rollout/evaluation design; its performance remains unverified until run on the CUDA machine. The existing AMD notebooks and historical Pendulum notebook remain untouched.
