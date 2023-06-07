# Basic imports
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Custom imports
from utils import *
import tf_idf
from metrics import Metric

def main():
    # Set random seed
    RANDOM_STATE = 1989

    parent_directory = os.path.abspath(os.path.join(os.getcwd(), ".."))
    file_path = os.path.join(parent_directory, "dataset/name_full.csv")

    # Import full data and split train-test set
    data = pd.read_csv(file_path)
    X = data["Full_Name"].values
    y = data["Gender"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    
    # Vectorize/Embed via TF-IDF
    TF_IDF = tf_idf.Compute_TF_IDF(X_train)
    tfidf_X_train = TF_IDF.compute_tf_idf()
    tfidf_X_test = TF_IDF.compute_tf_idf_for_test(X_test)

    # Reduce dimensionality via TruncatedSVD
    tfidf_X_train, tfidf_X_test = truncated_svd(tfidf_X_train, tfidf_X_test, random_state=RANDOM_STATE)
    
    # Apply different models
    ## LOGREG
    
    LOGREG_PARAM_GRID = {
        'penalty': [None, "l1", "l2"],
        'C': [0.01, 0.03, 0.1, 0.3, 1, 3]
    }
    logreg = MyLogisticRegression(random_state=RANDOM_STATE,
                                grid_search=True, scoring="accuracy", param_grid=LOGREG_PARAM_GRID, cv=5)
    logreg.fit(tfidf_X_train, y_train)
    y_pred = logreg.predict(tfidf_X_test)
    y_proba = logreg.predict_proba(tfidf_X_test)
    logreg_metric = Metric(y_test, y_pred, y_proba[:, 1])
    
    pr_disp = logreg_metric.getRocCurveClass1()
    pr_disp.plot()
    plt.show()

if __name__ == "__main__":
    main()