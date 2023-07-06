"""
    Created by @namhainguyen2803 on 06/07/2023
"""

import main
import numpy as np

input_name = input("Input name to predict: ")
name = [input_name]
encode_name = main.TF_IDF.compute_tf_idf_for_test(name)
x_test = main.svd.transform(encode_name)
print(f"Test set information: {x_test.shape}")
if main.name_model == "logistic":

    for i in range(len(name)):
        proba = main.model.predict_proba(np.reshape(x_test[i], (1, -1)), True)[0]
        if proba > 0.5:
            gender = "male"
            print(f"Name: {name[i]}, decision: {gender}, with probability: {proba}")
        else:
            gender = "female"
            print(f"Name: {name[i]}, decision: {gender}, with probability: {1 - proba}")

elif main.name_model == "svm":

    for i in range(len(name)):
        proba = main.model.predict_class(x_test[i])
        if proba == 1:
            gender = "male"
        else:
            gender = "female"
        print(f"Name: {name[i]}, decision: {gender}")

elif main.name_model == "nb":

    for i in range(len(name)):
        proba = main.model.predict(encode_name[i])[0]
        if proba > 0.5:
            gender = "male"
            print(f"Name: {name[i]}, decision: {gender}, with probability: {proba}")
        else:
            gender = "female"
            print(f"Name: {name[i]}, decision: {gender}, with probability: {1 - proba}")

elif main.name_model == "nn":

    for i in range(len(name)):
        proba = main.model.predict_proba(np.reshape(x_test[i], (1, -1)))[0][0]
        if proba > 0.5:
            gender = "male"
            print(f"Name: {name[i]}, decision: {gender}, with probability: {proba}")
        else:
            gender = "female"
            print(f"Name: {name[i]}, decision: {gender}, with probability: {1 - proba}")
