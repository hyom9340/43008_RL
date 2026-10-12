# Reinforcement Learning Assignment 2

The first version of the project is collected in [`original/`](original/). Its notebooks, generator scripts, dependencies and setup notes have been preserved; `main` remains unchanged.

The new NVIDIA CUDA workflow is in [`submission_cuda/`](submission_cuda/). Run `build_cuda_submission.py` from the repository root to regenerate its three notebooks. Breakout PPO uses a revised rollout/evaluation design; improvement remains unverified until the CUDA notebook is run.

Open the repository in VS Code and select the appropriate kernel for the notebook you are running. For the original local notebooks, install from `original/requirements.txt`. For the CUDA notebooks, see [`submission_cuda/README.md`](submission_cuda/README.md) and `requirements-cuda.txt`.

Training outputs, model checkpoints, MP4s, local virtual environments and submission ZIP files are excluded from Git. Keep them locally or in a separate backup. Save the executed `.ipynb` files with their cell outputs before submission.
