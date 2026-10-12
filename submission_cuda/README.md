# Assignment 2 CUDA run and validation / 作业 2 CUDA 运行与验证

This is the runbook for the **`assignment2/cuda-notebook-redesign` branch**. The notebooks are training code, not proof that training has already completed.

本文用于 **`assignment2/cuda-notebook-redesign` 分支**。Notebook 中的训练目标只是代码设置，不能当作已完成的训练结果。

## 1. Files and order / 文件与运行顺序

Run from the repository root, top to bottom, in separate notebook kernels. Do not combine historical output files with a new run.

在仓库根目录打开项目；三本 notebook 分别选择内核、从上到下运行。不要把历史输出文件与新实验混用。

| Notebook | Purpose / 用途 | Expected new evidence / 应产生的新证据 |
| --- | --- | --- |
| `Breakout_DQN_PPO_CUDA.ipynb` | Custom CNN + DQN/PPO / 自定义 CNN 与两种算法 | Training logs, periodic and final evaluations, figures, two MP4s / 训练日志、阶段与最终评估、图、两个 MP4 |
| `CartPole_DQN_PPO_CUDA.ipynb` | Custom ANN + DQN/PPO / 自定义 ANN 与两种算法 | Separate episode counts, evaluations, figures, two MP4s / 独立回合数、评估、图、两个 MP4 |
| `Pendulum_Q_Learning.ipynb` | Discretised TD Q-Learning / 离散化 TD Q-Learning | Q-table, curve, baseline comparison, one MP4 / Q 表、曲线、随机基线对比、一个 MP4 |

`build_cuda_submission.py` regenerates all three notebooks and **overwrites their saved cell outputs**. Run it only before training, never after you have executed and saved the final notebooks.

`build_cuda_submission.py` 会重新生成三本 notebook，**覆盖其中已保存的运行输出**。只在训练前运行，不要在最终运行并保存后再次运行。

## 2. Before training / 训练前检查

- [ ] Select the NVIDIA CUDA Python/Jupyter kernel. The first cell must print the intended Python executable, PyTorch CUDA version, and GPU name; `torch.cuda.is_available()` must be `True`. / 选择 NVIDIA CUDA 内核；首格应打印正确的 Python 路径、PyTorch CUDA 版本和显卡名称，且 CUDA 可用。
- [ ] Keep the working CUDA-enabled PyTorch installation. Install missing packages from `requirements-cuda.txt` into that environment, then check CUDA availability again. / 保留当前可用的 CUDA PyTorch；在同一环境安装缺失依赖后，再次检查 CUDA 是否可用。
- [ ] Confirm `ALE/Breakout-v5` can be created and CartPole/Pendulum open successfully. / 确认 Breakout ROM/环境、CartPole、Pendulum 均能创建。
- [ ] Confirm the notebook working directory is the repository root. New Breakout/CartPole artifacts are written under `submission_cuda_outputs/`; Pendulum writes to `pendulum_artifacts/`. / 确认工作目录是仓库根目录；前两本输出到 `submission_cuda_outputs/`，Pendulum 输出到 `pendulum_artifacts/`。
- [ ] Allow enough disk space for DQN replay-buffer checkpoints and MP4s. Keep model ZIP **and** replay PKL together if you plan to resume DQN. / 为 DQN 回放缓存和视频预留空间；需要续训 DQN 时，同时保留模型 ZIP 与 replay PKL。
- [ ] Start with a fresh `submission_cuda_outputs/` directory for a new experiment. If you already have a genuine interrupted run, keep its matching logs and checkpoints together. Never place artifacts from the earlier submission ZIP in this directory. / 新实验使用全新的输出目录；若确实要续训，日志和检查点必须来自同一次实验，不要放入旧 ZIP 的结果。

Quick terminal check in the selected CUDA environment / 在选定 CUDA 环境中的快速检查：

```powershell
python -c "import sys, torch; print(sys.executable, torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); assert torch.cuda.is_available()"
python -c "import gymnasium as gym, ale_py; gym.register_envs(ale_py); e=gym.make('ALE/Breakout-v5'); print(e.observation_space, e.action_space); e.close()"
```

The source notebook used as a reference had machine-specific Python and output paths. These new notebooks use the selected kernel and `Path.cwd()`; do not copy those old absolute paths back in.

参考 notebook 写死了另一台机器的 Python 和输出路径；新版使用当前内核及 `Path.cwd()`，不要再复制旧的绝对路径。

## 3. Required training evidence / 必须核对的训练证据

| Experiment / 实验 | Code target / 代码目标 | Pass check / 通过条件 |
| --- | ---: | --- |
| Breakout DQN | 3,600,000 steps | Actual completed steps ≥ 1,000,000; final model, replay buffer, logs and evaluation exist / 实际完成至少 100 万步，并有模型、回放缓存、日志及评估 |
| Breakout PPO | 6,000,000 steps | Actual completed steps ≥ 1,000,000; a best checkpoint selected after 1,000,000 steps exists / 实际完成至少 100 万步，且存在达到门槛后选出的最佳检查点 |
| CartPole DQN | 10,000 episodes | Logged completed episodes ≥ 1,000 / 实际记录的完整回合至少 1,000 |
| CartPole PPO | 10,000 episodes | Logged completed episodes ≥ 1,000 / 实际记录的完整回合至少 1,000 |
| Pendulum Q-Learning | 50,000 episodes in copied notebook | Executed episode count ≥ 1,000 and Q-table/curve/evaluation exist / 实际完成至少 1,000 回合，并有 Q 表、曲线与评估 |

The Atari **one-million-time-step** interpretation comes from the course clarification supplied with the assignment; the PDF wording alone says “1,000,000 for Atari.” The target values above exceed the minimum but are not themselves execution evidence.

Atari **100 万时间步**的口径依据课程补充说明；PDF 本身写的是“1,000,000 for Atari”。上表目标高于最低要求，但配置值不能代替实际运行输出。

## 4. Breakout PPO change to verify / Breakout PPO 改动的验证

Previous run: one environment, `n_steps=128`, `batch_size=128`, fixed learning rate `2.5e-4`; its 30-episode final evaluation was **13.43 ± 3.90**. This is historical context, not a directly comparable control because the new evaluation uses specified fixed seeds.

旧实验使用单环境、`n_steps=128`、`batch_size=128`、固定学习率 `2.5e-4`，30 回合最终评估为 **13.43 ± 3.90**。这只是历史背景；新版采用固定种子，不能直接拿两个数字宣称提升。

New proposal: `n_steps=2048`, `batch_size=256`, four epochs, learning-rate decay, evaluation every 200,000 steps on ten fixed seeds, and best-model selection only after one million training steps. These are hypotheses to test, **not proven causes or guaranteed improvements**.

新方案：`n_steps=2048`、`batch_size=256`、四轮更新、学习率递减、每 20 万步用十个固定种子评估，并且只在训练满 100 万步后选最佳模型。这些是待验证的方案，**不是已证实的原因或效果保证**。

- [ ] Check that the periodic **raw-reward** evaluation curve trends upward or stabilises, and inspect dips rather than quoting only the best point. / 查看阶段评估的**原始奖励**曲线是否上升或稳定，也要分析下降区间，不能只报最高点。
- [ ] Check the 30-seed final PPO result against the **new-run** random baseline and DQN using the same environment, seeds, reward units, and deterministic policy. / 在同一环境、种子、奖励单位和确定性策略下，将新版 PPO 的 30 回合结果与新版随机基线及 DQN 比较。
- [ ] If claiming improvement over the old PPO, re-evaluate the old checkpoint under the **same new protocol**. If unavailable, state that the historical 13.43 result is not a controlled comparison. / 若要声称优于旧 PPO，必须用**同一套新评估流程**重评旧模型；做不到就明确历史的 13.43 不是严格对照。
- [ ] If PPO is still weak, report the result honestly. A suggested follow-up is a controlled run changing **one** hyperparameter group at a time, not a claim that more steps will necessarily solve it. / 若 PPO 仍弱，应如实报告；后续对照实验应每次只调整一组参数，而不能断言增加训练步数一定有效。

Training Breakout uses **clipped rewards**, while evaluation and random baseline use **raw rewards**. Do not subtract or overlay these values as if they share a unit. New Monitor logs are append-only for a matching resumed run; inspect for accidental duplicates or mixed runs before plotting.

Breakout 训练使用**裁剪奖励**，评估与随机基线使用**原始奖励**。不要把两者当同一单位直接相减或叠图。续训日志只应追加同一次实验的数据；画图前检查是否混入重复或别的运行。

## 5. Figures, evaluation and MP4 / 图表、评估与视频

- [ ] Both Breakout and CartPole show separate DQN/PPO training curves, with a labelled axis, raw points and moving average. Breakout additionally shows fixed-seed evaluation over training steps. / Breakout 与 CartPole 分别展示 DQN/PPO 训练曲线，标清坐标、原始点及移动平均；Breakout 另展示按训练步数排列的固定种子评估曲线。
- [ ] Final comparison tables/figures use freshly reloaded saved models and episode-level results; report mean, standard deviation and episode count. The report numbers must match the notebook CSVs. / 最终对比表和图基于重新加载的模型及逐回合结果，写明均值、标准差和回合数；报告数据须与 notebook CSV 一致。
- [ ] Verify the custom CNN/ANN printed in the notebook is the one actually used by the trained policies. Match any report network diagrams to these printed layers and shapes. / 核实 notebook 打印的自定义 CNN/ANN 确实用于训练；报告网络图要与实际层及维度一致。
- [ ] Produce five MP4s in total: two Breakout, two CartPole, one Pendulum. Open each file and check visible movement and a complete episode. Note its model, seed, single-episode reward and length. A video score is **not** the multi-episode evaluation mean. / 共检查五个 MP4：Breakout 两个、CartPole 两个、Pendulum 一个。逐个打开确认画面、完整回合，并记录模型、种子、单局奖励与长度；视频单局分数**不是**多局评估均值。
- [ ] Save notebooks after all cells finish. Breakout/CartPole notebooks in Git are **unexecuted templates** until you run them on CUDA; the copied Pendulum notebook contains historical outputs that must be replaced if rerun. / 所有单元运行完毕后保存 notebook；Git 中 Breakout/CartPole 目前是**未运行模板**。Pendulum 复制本带有历史输出，重新运行后要保存新的输出。

## 6. Final handoff / 最终提交前

- [ ] Reopen each saved `.ipynb` and confirm outputs are visible, no cells show errors, and the last artifact-check cell passes. / 重新打开保存后的 notebook，确认输出可见、无错误，末尾文件检查通过。
- [ ] Back up the ignored `submission_cuda_outputs/` and `pendulum_artifacts/` folders; Git push does not upload the models, CSVs or MP4s. / 备份被 Git 忽略的两个输出目录；推送 Git 不会上传模型、CSV 或 MP4。
- [ ] Package the notebook ZIP, report PDF (10-page maximum), AI-use declaration and requested simulation video(s) according to Canvas instructions. Download the submitted files once and verify them. / 按 Canvas 要求提交 notebook ZIP、最多十页的 PDF 报告、AI 使用声明及所需模拟视频；提交后下载检查。

For the assignment wording and marking details, use the course specification and Canvas announcements as the authority. This README is an execution checklist, not a replacement for them.

作业文字和评分要求以课程规范及 Canvas 公告为准。本 README 是运行核对清单，不能代替课程说明。
