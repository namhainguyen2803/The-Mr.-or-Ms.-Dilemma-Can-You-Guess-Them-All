# Basic imports
import matplotlib.pyplot as plt

# Custom imports
from models import *
from data_reader import *
from result_exporter import *

def main():
    # Set random seed
    RANDOM_STATE = 1989 
    DROP_DUP = True
    TRUNCATE = True
    COMPONENT_TEST = None # None, l, m, f, mf - corresponding to full, last, middle, first, middle-first names

    # Read data
    if COMPONENT_TEST == None:
        tfidf_X_train, tfidf_X_test, y_train, y_test = data_reader(random_state=RANDOM_STATE, drop_dup=DROP_DUP, truncate=TRUNCATE)
    else:
        tfidf_X_train, tfidf_X_test, y_train, y_test = component_data_reader(random_state=RANDOM_STATE, component_test=COMPONENT_TEST)
    
    # Apply different models
    
        ## LOGREG 
    
    LOGREG_PARAM_GRID = [
        {
            'penalty': [None]
        },
        {
            'penalty': ["l1", "l2"],
            'C': [0.01, 0.03, 0.1, 0.3, 1, 3]
        },
        {
            'penalty': ["elasticnet"],
            'C': [0.01, 0.03, 0.1, 0.3, 1, 3],
            'l1_ratio': [0.2, 0.4, 0.6, 0.8]
        }
    ]
    logreg = MyLogisticRegression(tfidf_X_train, tfidf_X_test, y_train, y_test, random_state=RANDOM_STATE,
                                grid_search=True, scoring="accuracy", param_grid=LOGREG_PARAM_GRID, cv=5)
    logreg.evaluate()
    export_result(model=logreg, random_state=RANDOM_STATE, drop_dup=DROP_DUP, truncate=TRUNCATE, component_test=COMPONENT_TEST)
    
        ## SVM
    
    SVM_PARAM_GRID = {
        'C': [0.01, 0.03, 0.1, 0.3, 1, 3],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    }
    svm = MySupportVectorMachine(tfidf_X_train, tfidf_X_test, y_train, y_test, random_state=RANDOM_STATE,
                                 grid_search=True, scoring="accuracy", param_grid=SVM_PARAM_GRID, cv=5)
    svm.evaluate()
    export_result(model=svm, random_state=RANDOM_STATE, drop_dup=DROP_DUP, truncate=TRUNCATE, component_test=COMPONENT_TEST)
    
        ## KNN
    
    KNN_PARAM_GRID = [
        {
            'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
            'weights': ['uniform', 'distance'],
            'metric': ['cityblock', 'cosine', 'euclidean']
        },
        {
            'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
            'weights': ['uniform', 'distance'],
            'p': [1, 2, 3, 4, 5],
            'metric': ['minkowski']
        }
    ]
    knn = MyKNearestNeighbor(tfidf_X_train, tfidf_X_test, y_train, y_test,
                             grid_search=True, scoring="accuracy", param_grid=KNN_PARAM_GRID, cv=5)
    knn.evaluate()
    export_result(model=svm, random_state=RANDOM_STATE, drop_dup=DROP_DUP, truncate=TRUNCATE, component_test=COMPONENT_TEST)
    
        ## GNB
    
    gnb = MyGaussianNaiveBayes(tfidf_X_train, tfidf_X_test, y_train, y_test)
    gnb.evaluate()
    export_result(model=gnb, random_state=RANDOM_STATE, drop_dup=DROP_DUP, truncate=TRUNCATE, component_test=COMPONENT_TEST)
    
        ## MNB
    
    mnb = MyMultinomialNaiveBayes(tfidf_X_train, tfidf_X_test, y_train, y_test)
    mnb.evaluate()
    export_result(model=mnb, random_state=RANDOM_STATE, drop_dup=DROP_DUP, truncate=TRUNCATE, component_test=COMPONENT_TEST)
    
        ## BNB
    
    bnb = MyBernoulliNaiveBayes(tfidf_X_train, tfidf_X_test, y_train, y_test)
    bnb.evaluate()
    export_result(model=bnb, random_state=RANDOM_STATE, drop_dup=DROP_DUP, truncate=TRUNCATE, component_test=COMPONENT_TEST)
    
        ### DT
    
    DT_PARAM_GRID = {
        'criterion': ["entropy"],
        'max_depth': [20, 40, 60],
        'max_leaf_nodes': [50, 75, 100],
        'ccp_alpha': [0.01, 0.03, 0.1]
    }
    dt = MyDecisionTree(tfidf_X_train, tfidf_X_test, y_train, y_test, random_state=RANDOM_STATE,
                        grid_search=True, scoring="accuracy", param_grid=DT_PARAM_GRID, cv=5)
    dt.evaluate()
    export_result(model=dt, random_state=RANDOM_STATE, drop_dup=DROP_DUP, truncate=TRUNCATE, component_test=COMPONENT_TEST)
    
        ### RF
    
    RF_PARAM_DRIG = [
        {
            'n_estimators': [50, 75, 100, 125],
            'criterion': ["entropy"],
            'max_depth': [20, 40, 60],
            'max_leaf_nodes': [50, 75, 100],
            'ccp_alpha': [0.003, 0.01, 0.03],
            'bootstrap': [False],
        },
        {
            'n_estimators': [50, 75, 100, 125],
            'criterion': ["entropy"],
            'max_depth': [20, 40, 60],
            'max_leaf_nodes': [50, 75, 100],
            'ccp_alpha': [0.003, 0.01, 0.03],
            'bootstrap': [True],
            'oob_score': [True, False],
            'max_samples': [0.25, 0.5, 0.75, 1]
        }
    ]
    rf = MyRandomForest(tfidf_X_train, tfidf_X_test, y_train, y_test, random_state=RANDOM_STATE,
                        grid_search=True, scoring="accuracy", param_grid = RF_PARAM_DRIG, cv=5)
    rf.evaluate()
    export_result(model=rf, random_state=RANDOM_STATE, drop_dup=DROP_DUP, truncate=TRUNCATE, component_test=COMPONENT_TEST)
    
    # After training, with fixed hyperparameters for each model

        ## LOGREG 
    
    LOGREG_PARAM_GRID = {
        'penalty': [None]
    }
    logreg = MyLogisticRegression(tfidf_X_train, tfidf_X_test, y_train, y_test, random_state=RANDOM_STATE,
                                grid_search=True, scoring="accuracy", param_grid=LOGREG_PARAM_GRID, cv=5)
    logreg.evaluate()
    export_result(model=logreg, random_state=RANDOM_STATE, drop_dup=DROP_DUP, truncate=TRUNCATE, component_test=COMPONENT_TEST)
    
        ## SVM
    
    SVM_PARAM_GRID = {
        'C': [1],
        'kernel': ['linear']
    }
    svm = MySupportVectorMachine(tfidf_X_train, tfidf_X_test, y_train, y_test, random_state=RANDOM_STATE,
                                 grid_search=True, scoring="accuracy", param_grid=SVM_PARAM_GRID, cv=5)
    svm.evaluate()
    export_result(model=svm, random_state=RANDOM_STATE, drop_dup=DROP_DUP, truncate=TRUNCATE, component_test=COMPONENT_TEST)
    
        ## BNB
    
    bnb = MyBernoulliNaiveBayes(tfidf_X_train, tfidf_X_test, y_train, y_test)
    bnb.evaluate()
    export_result(model=bnb, random_state=RANDOM_STATE, drop_dup=DROP_DUP, truncate=TRUNCATE, component_test=COMPONENT_TEST)
    
        ### RF
    
    RF_PARAM_DRIG = {
        'n_estimators': [100],
        'criterion': ["entropy"],
        'max_depth': [20],
        'max_leaf_nodes': [75],
        'ccp_alpha': [0.003],
        'bootstrap': [True],
        'oob_score': [True],
        'max_samples': [0.25]
    }
    rf = MyRandomForest(tfidf_X_train, tfidf_X_test, y_train, y_test, random_state=RANDOM_STATE,
                        grid_search=True, scoring="accuracy", param_grid = RF_PARAM_DRIG, cv=5)
    rf.evaluate()
    export_result(model=rf, random_state=RANDOM_STATE, drop_dup=DROP_DUP, truncate=TRUNCATE, component_test=COMPONENT_TEST)
    
if __name__ == "__main__":
    main()