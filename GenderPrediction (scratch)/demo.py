"""
    Created by @namhainguyen2803 on 05/07/2023
"""

import main
import pandas as pd
import numpy as np

LINK_DATA = "dataset/dsai-k66.csv"
df = pd.read_csv(LINK_DATA)
full_data = df["Full_Name"].to_numpy()
full_label = df["Gender"].to_numpy()
# name = ["Nguyen Nam Hai"]
encode_name = main.TF_IDF.compute_tf_idf_for_test(full_data)
x_test = main.svd.transform(encode_name)
print(f"Test set information: {x_test.shape}")
if main.name_model == "logistic":
    print(main.model.compute_accuracy(x_test, full_label, True))
    for i in range(len(full_label)):
        proba = main.model.predict_proba(np.reshape(x_test[i], (1, -1)), True)[0]
        if proba > 0.5:
            gender = "male"
            print(f"Name: {full_data[i]}, decision: {gender}, with probability: {proba}")
        else:
            gender = "female"
            print(f"Name: {full_data[i]}, decision: {gender}, with probability: {1 - proba}")

elif main.name_model == "svm":
    y_test = np.zeros_like(full_label)
    y_test[full_label == 1] = 1
    y_test[full_label == 0] = -1
    print(main.model.score(x_test, y_test))
    for i in range(len(full_label)):
        proba = main.model.predict_class(x_test[i])
        if proba == 1:
            gender = "male"
        else:
            gender = "female"
        print(f"Name: {full_data[i]}, decision: {gender}")

elif main.name_model == "nb":
    encode_name[encode_name != 0] = 1
    print(main.model.test_accuracy(encode_name, full_label))
    for i in range(len(full_label)):
        proba = main.model.predict(encode_name[i])[0]
        if proba > 0.5:
            gender = "male"
            print(f"Name: {full_data[i]}, decision: {gender}, with probability: {proba}")
        else:
            gender = "female"
            print(f"Name: {full_data[i]}, decision: {gender}, with probability: {1 - proba}")

elif main.name_model == "nn":
    print(main.model.eval(x_test, full_label))
    for i in range(len(full_label)):
        proba = main.model.predict_proba(np.reshape(x_test[i], (1, -1)))[0][0]
        if proba > 0.5:
            gender = "male"
            print(f"Name: {full_data[i]}, decision: {gender}, with probability: {proba}")
        else:
            gender = "female"
            print(f"Name: {full_data[i]}, decision: {gender}, with probability: {1 - proba}")