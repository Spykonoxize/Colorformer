# Project Structure

This repository follows a modular Python layout for reusable code.

- `src/colorformer/`: Core modular package (dataset, model, trainer, losses, inference)
- `scripts/`: Training and inference entry points
- `data/`: Local-only datasets (ignored by git)
- `outputs/`: Generated artifacts such as checkpoints and figures (ignored by git)
- `docs/`: Project notes and documentation
