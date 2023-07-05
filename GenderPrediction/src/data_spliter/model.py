"""
    Created by @namhainguyen2803 on 29/06/2023
"""

import numpy as np
import pandas as pd

class DataSpliter(object):

    def __init__(self, link_data, split_ratio, normalize=False):
        self.link_data = link_data
        self.split_ratio = np.array(split_ratio)
        self.normalize = normalize

        self.eps = 1e-5
        assert len(self.split_ratio) == 2 or len(self.split_ratio) == 3, "length of split_ratio must be in 2 or 3."
        assert 1 - self.eps < np.sum(self.split_ratio) < 1 + self.eps, "split ratio array must sum to 1"
        if len(self.split_ratio) == 2:
            self.having_validation = False
        else:
            self.having_validation = True

    def split(self):
        df = pd.read_csv(self.link_data)
        full_data = df["Full_Name"].to_numpy()
        full_label = df["Gender"].to_numpy()
        assert np.shape(full_data)[0] == np.shape(full_label)[0], "shape between data and label must be the same."

        shuffle_index = np.arange(np.shape(full_data)[0])
        np.random.shuffle(shuffle_index)
        full_data = full_data[shuffle_index]
        full_label = full_label[shuffle_index]

        candidate = np.random.uniform(low=0, high=1, size=np.shape(full_data))
        decision = np.zeros_like(candidate)

        decision[candidate > self.split_ratio[0]] = 1
        if self.having_validation:
            decision[candidate > (self.split_ratio[0] + self.split_ratio[1])] = 2

        training_set = {"X": full_data[decision == 0],
                        "y": full_label[decision == 0]}
        test_set = {"X": full_data[decision == 1],
                    "y": full_label[decision == 1]}

        if self.having_validation:
            validation_set = {"X": full_data[decision == 2],
                              "y": full_label[decision == 2]}
            data = {"train": training_set, "test": test_set, "validation": validation_set}
            return data
        else:
            data = {"train": training_set, "test": test_set}
            return data


if __name__ == "__main__":
    data_path = "dataset/name_full.csv"
    split_ratio = [0.6, 0.2, 0.2]
    dataloader = DataSpliter(data_path, split_ratio)
    print(dataloader.split())
