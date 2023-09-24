# Chess Embedding Experiments

## Cheat Sheet

- Generate random training positions

    python -m src.run.generate_random_positions

- Train a neural network

    python -m src.run.train

- Evaluate a trained network by starting the notebook: `src/run/evaluate.ipynb`

## Tools

- Start ML Flow UI, in correct python venv

    mlflow ui

- Export dependencies to requirements.txt

    poetry export > requirements.txt
