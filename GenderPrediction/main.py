"""
    Created by @namhainguyen2803 on 29/05/2023
"""

import numpy as np
import pandas as pd
import src.truncatedSVD.model as truncatedsvd
import src.logistic.model as logistic
import src.nn.architecture as architecture
import src.nn.model as nn
import src.nn.optimization as optim
import src.svm.model as svm
import src.tfidf.tf_idf as tf_idf
import src.naivebayes.model as nb
import src.data_spliter.model as split
import time

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC

# hyperparameters
np.random.seed(2803)
LINK_FULL = "dataset/name_full.csv"
SPLIT_RATIO = [0.7, 0.3]
spliter = split.DataSpliter(LINK_FULL, SPLIT_RATIO)

data = spliter.split()

training_name = data["train"]["X"]
y = data["train"]["y"]

test_name = data["test"]["X"]
y_test = data["test"]["y"]

# TFIDF
TF_IDF = tf_idf.Compute_TF_IDF(training_name)
training_tf_idf_matrix = TF_IDF.compute_tf_idf()
test_tf_idf_matrix = TF_IDF.compute_tf_idf_for_test(test_name)

# Truncated SVD
svd = truncatedsvd.TruncatedSVD(training_tf_idf_matrix, 100)
X = svd.fit()
X_test = svd.transform(test_tf_idf_matrix)

name_model = input("Model to predict [svm (Support Vector Machine), logistic (Logistic Regression), nb (Naive Bayes), nn (Neural Network)]: ")
assert name_model in ("svm", "logistic", "nb", "nn"), "Name of model is undefined"
print(f"Training set information: {X.shape, y.shape}")

if name_model == "svm":

    y[y == 0] = -1
    y_test[y_test == 0] = -1

    model = svm.Support_Vector_Machine(X, y, 1)
    start = time.time()
    model.fit(20)
    end = time.time()
    print(f"Test set information: {X_test.shape, y_test.shape}")
    score = model.score(X_test, y_test)
    print(f"Accuracy on test set: {score}, Training time: {end - start}")

    start_lib = time.time()
    clf = SVC(C=1, kernel="linear", max_iter=50).fit(X, y)
    end_lib = time.time()
    acc_lib = clf.score(X_test, y_test)
    print(f"Accuracy of library on test set: {acc_lib}, Training time: {end_lib - start_lib}")

elif name_model == "logistic":
    # train
    model = logistic.LogisticRegression(X, y, penalty="l2")
    start = time.time()
    model.fit()
    end = time.time()
    # test
    print(f"Test set information: {X_test.shape, y_test.shape}")
    test_accuracy = model.compute_accuracy(X_test, y_test, True)
    print(f"Accuracy on test set: {test_accuracy}, Training time: {end - start}")

    start_lib = time.time()
    clf = LogisticRegression(penalty="l2", C=1e5, max_iter=500).fit(X, y)
    end_lib = time.time()
    acc_lib = clf.score(X_test, y_test)
    print(f"Accuracy of library on test set: {acc_lib}, Training time: {end_lib - start_lib}")

elif name_model == "nb":
    training_tf_idf_matrix[training_tf_idf_matrix != 0] = 1
    test_tf_idf_matrix[test_tf_idf_matrix != 0] = 1
    model = nb.NaiveBayes(training_tf_idf_matrix, y)
    start = time.time()
    model.fit()
    end = time.time()
    print(f"Test set information: {test_tf_idf_matrix.shape, y_test.shape}")
    acc = model.test_accuracy(test_tf_idf_matrix, y_test)
    print(f"Accuracy on test set: {acc}, Training time: {end - start}")

    start_lib = time.time()
    clf = BernoulliNB(force_alpha=True).fit(training_tf_idf_matrix, y)
    end_lib = time.time()
    acc_lib = clf.score(test_tf_idf_matrix, y_test)
    print(f"Accuracy of library on test set: {acc_lib}, Training time: {end_lib - start_lib}")

elif name_model == "nn":
    architecture = architecture.FullyConnectedNet(input_dim=100, hidden_dims=[256, 64, 16, 8])
    optimizer = optim.rmsprop

    hyper_params = {
        "num_epochs": 300,
        "batch_size": 1000,
        "lr_decay": 0.5,
        "learning_rate": 0.001,
        "verbose": True,
        "print_every": 10,
        "regularization": 0.01
    }
    model = nn.VanillaFeedForwardNetwork(architecture=architecture, X_train=X, y_train=y, optim=optimizer, **hyper_params)
    start = time.time()
    model.train()
    finish = time.time()

    # test
    print(f"Test set information: {test_tf_idf_matrix.shape, y_test.shape}")
    test_acc = model.eval(X_test, y_test)
    print(f"Accuracy on test set: {test_acc}, Training time: {finish - start}")
