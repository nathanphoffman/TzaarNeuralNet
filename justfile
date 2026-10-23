# Run `just` to list the commands.

default:
    @just --list

# Extra flags pass through, e.g. `just train --generations 20` or `just train --fresh`.
# Train the Tzaar model until Ctrl+C (resumes from the last saved generation)
train *args:
    cd tzaar && uv run train.py {{args}}

# Play against the trained model in your browser (opens at http://localhost:8001)
play:
    cd tzaar && uv run serve.py
