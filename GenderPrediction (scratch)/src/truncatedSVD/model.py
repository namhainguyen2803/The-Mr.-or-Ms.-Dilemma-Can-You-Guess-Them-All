"""
    Created by @namhainguyen2803 on 03/06/2023
"""

import numpy as np

class TruncatedSVD():
    def __init__(self, matrix, num_component):
        self.matrix = matrix
        self.num_component = num_component
        self.U = None
        self.S = None
        self.VT = None
    def fit(self):
        U, S, VT = np.linalg.svd(self.matrix, full_matrices=False)
        maximum_numComponents = len(S)
        assert maximum_numComponents > self.num_component, "Number of components exceeds the maximum number of singular values"
        self.U = U[:, :self.num_component]
        self.S = S[:self.num_component]
        self.VT = VT[:self.num_component, :]
        return self.U.dot(np.diag(self.S))
    def transform(self, X_test):
        return X_test.dot(self.VT.T)

if __name__ == "__main__":
    mat = np.random.rand(10, 4)
    model = TruncatedSVD(mat, 2)
    model.fit()
    test_mat = np.random.rand(5, 4)
    print(model.transform(test_mat))