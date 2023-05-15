"""
    Created by @namhainguyen2803 in 10/05/2023
"""

import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, matrix):
        self.matrix = matrix

    def __len__(self):
        return len(self.matrix)

    def __getitem__(self, idx):
        return self.matrix[idx]
