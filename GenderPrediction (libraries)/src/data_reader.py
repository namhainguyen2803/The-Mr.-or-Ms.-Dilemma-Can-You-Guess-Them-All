import os
import pandas as pd
from tf_idf import *
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD

def truncated_svd(X_train, X_test, random_state, n_components=100):
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    svd.fit(X_train)
    transformed_X_train = svd.transform(X_train)
    transformed_X_test = svd.transform(X_test)
    return transformed_X_train, transformed_X_test

def data_reader(random_state, drop_dup=True, truncate=True):
    parent_directory = os.path.abspath(os.path.join(os.getcwd(), ".."))
    if drop_dup == True:
        file_path = os.path.join(parent_directory, "dataset/processed-full-name.csv")
    else:
        file_path = os.path.join(parent_directory, "dataset/processed-full-name-dup.csv")

    # Import full data and split train-test set
    data = pd.read_csv(file_path)
    X = data["Full_Name"].values
    y = data["Gender"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
    
    # Vectorize/Embed via TF-IDF
    TF_IDF = Compute_TF_IDF(X_train)
    tfidf_X_train = TF_IDF.compute_tf_idf()
    tfidf_X_test = TF_IDF.compute_tf_idf_for_test(X_test)

    # Reduce dimensionality via TruncatedSVD
    if truncate == True:
        tfidf_X_train, tfidf_X_test = truncated_svd(tfidf_X_train, tfidf_X_test, random_state=random_state)
    
    return tfidf_X_train, tfidf_X_test, y_train, y_test

# DROP_DUP = FALSE, TRUNCATE = TRUE
def component_data_reader(random_state, component_test):
    parent_directory = os.path.abspath(os.path.join(os.getcwd(), ".."))
    if component_test == 'f': # first name
        file_path = os.path.join(parent_directory, "dataset/processed-first-name.csv")
    elif component_test == 'm': # middle name
        file_path = os.path.join(parent_directory, "dataset/processed-middle-name.csv")
    elif component_test == 'l': # last name
        file_path = os.path.join(parent_directory, "dataset/processed-last-name.csv")
    elif component_test == 'mf': # middle-first
        file_path = os.path.join(parent_directory, "dataset/processed-middle-first-name.csv")
    
    data = pd.read_csv(file_path)
    X = data["Full_Name"].values
    y = data["Gender"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

    TF_IDF = Compute_TF_IDF(X_train)
    tfidf_X_train = TF_IDF.compute_tf_idf()
    tfidf_X_test = TF_IDF.compute_tf_idf_for_test(X_test)

    tfidf_X_train, tfidf_X_test = truncated_svd(tfidf_X_train, tfidf_X_test, random_state=random_state)

    return tfidf_X_train, tfidf_X_test, y_train, y_test