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

training_path = "dataset/name_train.csv"
dev_path = "dataset/name_dev.csv"
test_path = "dataset/name_test.csv"
data_path = "dataset/name_full.csv"
df = pd.read_csv(training_path)
df_test = pd.read_csv(test_path)

full_data = df["Full_Names"]
TF_IDF = tf_idf.Compute_TF_IDF(full_data)
tf_idf_matrix = TF_IDF.compute_tf_idf()

test_data = df_test["Full_Names"]
test_tf_idf_matrix = TF_IDF.compute_tf_idf_for_test(test_data)

svd = truncatedsvd.TruncatedSVD(tf_idf_matrix, 100)
X = svd.fit()
y = df["Gender"].copy().to_numpy()

X_test = svd.transform(test_tf_idf_matrix)
y_test = df_test["Gender"].copy().to_numpy()

name_model = input("Model to predict [svm (Support Vector Machine), logistic (Logistic Regression), nb (Naive Bayes), rf (Random Forest), nn (Neural Network)]: ")
assert name_model in ("svm", "logistic", "nb", "rf", "nn"), "Name of model is undefined"
print(f"Training set information: {X.shape, y.shape}")

if name_model == "svm":

    y[y == 0] = -1
    y_test[y_test == 0] = -1

    model = svm.Support_Vector_Machine(X, y, 1)
    model.fit()

    print(f"Test set information: {X_test.shape, y_test.shape}")
    y_pred = model.predict_class(X_test)
    score = np.mean(y_pred == y_test)
    print(f"Accuracy on test set: {score}")

elif name_model == "logistic":
    # train
    model = logistic.LogisticRegression(X, y)
    model.fit()
    y_pred = model.predict_proba(X_test)

    # test
    logistic_score = 0
    print(f"Test set information: {X_test.shape, y_test.shape}")
    for i in range(len(X_test)):
        if y_pred[i].round() == y_test[i]:
            logistic_score += 1
    print(f"Accuracy on test set: {logistic_score / len(X_test)}")

elif name_model == "nb":
    tf_idf_matrix[tf_idf_matrix != 0] = 1
    test_tf_idf_matrix[test_tf_idf_matrix != 0] = 1
    model = nb.NaiveBayes(tf_idf_matrix, y)
    model.fit()
    print(f"Test set information: {test_tf_idf_matrix.shape, y_test.shape}")
    acc = model.test_accuracy(test_tf_idf_matrix, y_test)
    print(f"Accuracy on test set: {acc}")

elif name_model == "nn":
    architecture = architecture.FullyConnectedNet(input_dim=100, hidden_dims=[64, 16])
    optimizer = optim.rmsprop

    hyper_params = {
        "num_epochs": 50,
        "batch_size": 100,
        "lr_decay": 0.9,
        "learning_rate": 0.01,
        "verbose": True,
        "print_every": 10,
        "regularization": 0.01
    }
    model = nn.VanillaFeedForwardNetwork(architecture=architecture, X_train=X, y_train=y, X_test=X_test, y_test=y_test,
                                         optim=optimizer, **hyper_params)
    loss = model.train()

    # test
    acc = model.check_accuracy(X_test, y_test)
    print(f"Accuracy on test set: {acc}")
