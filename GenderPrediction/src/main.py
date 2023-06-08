# Basic imports
import matplotlib.pyplot as plt

# Custom imports
from models import *
from data_reader import *

def main():
    # Set random seed
    RANDOM_STATE = 1989

    # Read data
    tfidf_X_train, tfidf_X_test, y_train, y_test = data_reader(random_state=RANDOM_STATE, truncate=True)
    
    # Apply different models
    
        ## LOGREG
    """
    LOGREG_PARAM_GRID = {
        'penalty': [None, "l1", "l2"],
        'C': [0.01, 0.03, 0.1, 0.3, 1, 3]
    }
    logreg = MyLogisticRegression(tfidf_X_train, tfidf_X_test, y_train, y_test, random_state=RANDOM_STATE,
                                grid_search=True, scoring="accuracy", param_grid=LOGREG_PARAM_GRID, cv=5)
    logreg.evaluate()
    """
        ## SVM
    """
    SVM_PARAM_GRID = {
        'C': [0.01, 0.03, 0.1, 0.3, 1, 3],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    }
    svm = MySupportVectorMachine(tfidf_X_train, tfidf_X_test, y_train, y_test, random_state=RANDOM_STATE,
                                 grid_search=True, scoring="accuracy", param_grid=SVM_PARAM_GRID, cv=5)
    svm.evaluate()
    """
        ## KNN
    
    KNN_PARAM_GRID = {
        'n_neighbors': [7, 13, 19, 25],
        'weights': ['uniform', 'distance'],
        'p': [1, 2, 3, 4, 5],
        'metric': ['cityblock', 'cosine', 'euclidean', 'minkowski']
    }
    knn = MyKNearestNeighbor(tfidf_X_train, tfidf_X_test, y_train, y_test,
                             grid_search=True, scoring="accuracy", param_grid=KNN_PARAM_GRID, cv=5)
    knn.evaluate()
    

if __name__ == "__main__":
    main()