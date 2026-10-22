"""Train the net and save it. Run: uv run train.py"""
from tictactoe.model_store import save_net
from tictactoe.training import train_model

if __name__ == "__main__":
    save_net(train_model())
