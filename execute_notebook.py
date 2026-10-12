from pathlib import Path

import nbformat
from nbclient import NotebookClient


path = Path(__file__).resolve().parent / "Pendulum_TD_Q_Learning.ipynb"
with path.open("r", encoding="utf-8") as stream:
    notebook = nbformat.read(stream, as_version=4)

client = NotebookClient(
    notebook,
    timeout=1800,
    kernel_name="codex-rl",
    resources={"metadata": {"path": str(path.parent)}},
)
client.execute()

with path.open("w", encoding="utf-8") as stream:
    nbformat.write(notebook, stream)

print(f"Executed and saved {path}")
