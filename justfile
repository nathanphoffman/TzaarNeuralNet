# Run `just` to list the commands.

default:
    @just --list

# Extra flags pass through, e.g. `just train --generations 20` or `just train --models evolution-1`.
# Train all models side by side until Ctrl+C (resumes from the last saved generation)
train *args:
    uv run train.py {{args}}

# Play against any model or generation in your browser (opens at http://localhost:8001)
play:
    uv run serve.py

# Run the tests (they use a temporary models folder, never the real one)
test:
    uv run python -m unittest discover tests
