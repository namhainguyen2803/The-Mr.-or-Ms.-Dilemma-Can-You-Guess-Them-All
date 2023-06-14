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
    
    LOGREG_PARAM_GRID = {
        'penalty': [None, "l1", "l2", "elasticnet"],
        'C': [0.01, 0.03, 0.1, 0.3, 1, 3],
        'l1_ratio': [0.2, 0.4, 0.6, 0.8]
    }
    logreg = MyLogisticRegression(tfidf_X_train, tfidf_X_test, y_train, y_test, random_state=RANDOM_STATE,
                                grid_search=True, scoring="accuracy", param_grid=LOGREG_PARAM_GRID, cv=5)
    logreg.evaluate()
    
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
    """
    KNN_PARAM_GRID = {
        'n_neighbors': [7, 13, 19, 25],
        'weights': ['uniform', 'distance'],
        'p': [1, 2, 3, 4, 5],
        'metric': ['cityblock', 'cosine', 'euclidean', 'minkowski']
    }
    knn = MyKNearestNeighbor(tfidf_X_train, tfidf_X_test, y_train, y_test,
                             grid_search=True, scoring="accuracy", param_grid=KNN_PARAM_GRID, cv=5)
    knn.evaluate()
    """
        ## GNB
    """
    gnb = MyGaussianNaiveBayes(tfidf_X_train, tfidf_X_test, y_train, y_test)
    gnb.evaluate()
    """
        ## MNB
    """
    mnb = MyMultinomialNaiveBayes(tfidf_X_train, tfidf_X_test, y_train, y_test)
    mnb.evaluate()
    """
        ## BNB
    """
    bnb = MyBernoulliNaiveBayes(tfidf_X_train, tfidf_X_test, y_train, y_test)
    bnb.evaluate()
    """
        ### DT
    """
    DT_PARAM_GRID = {
        'criterion': ["entropy", "gini"],
        'max_depth': [20, 40, 60],
        'min_samples_split': [0.25, 0.5, 0.75, 1],
        'min_samples_leaf': [0.25, 0.5, 0.75, 1],
        'max_leaf_nodes': [20, 40, 60],
        'max_features': [0.5, 0.75],
        'ccp_alpha': [0.01, 0.03, 0.1]
    }
    dt = MyDecisionTree(tfidf_X_train, tfidf_X_test, y_train, y_test, random_state=RANDOM_STATE,
                        grid_search=True, scoring="accuracy", param_grid=DT_PARAM_GRID, cv=5)
    dt.evaluate()
    """
        ### RF
    """
    RF_PARAM_DRIG = {
        'n_estimators': [20, 40, 60],
        'criterion': ["entropy", "gini"],
        'max_depth': [20, 40, 60],
        'min_samples_split': [0.25, 0.5, 0.75, 1],
        'min_samples_leaf': [0.25, 0.5, 0.75, 1],
        'max_leaf_nodes': [20, 40, 60],
        'max_features': [0.25, 0.5, 0.75, 1],
        'ccp_alpha': [0.01, 0.03, 0.1, 0.3],
        'bootstrap': [True, False],
        'oob_score': [True, False],
        'max_samples': [0.25, 0.5, 0.75, 1]
    }
    rf = MyRandomForest(tfidf_X_train, tfidf_X_test, y_train, y_test, random_state=RANDOM_STATE,
                        grid_search=True, scoring="accuracy", param_grid = RF_PARAM_DRIG, cv=5)
    rf.evaluate()
    """

if __name__ == "__main__":
    main()