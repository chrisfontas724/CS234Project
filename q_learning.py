from models.grid import Grid
from qlearning.qstate import QState
from qlearning.mlp import MLPConfig, MLP
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn

def main():

    # Hardcode a simple grid for now.
    grid = Grid(filename="levels/grid_1.txt")

    # Create the MLP network with the configuration.
    mlp_config = MLPConfig(grid.size, grid.num_cols)
    mlp = MLP(mlp_config)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(mlp.parameters(), lr=0.001, momentum=0.9)

if __name__ == "__main__":
	main()