from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Visual purpose
import colorama
from colorama import Fore
colorama.init(autoreset=True)

def evaluate(model_info, y_true, y_pred):
    """
    Evaluate the model: accuracy, f1 score, confusion matrix
    """
    print(Fore.LIGHTYELLOW_EX + model_info)

    accuracy = accuracy_score(y_true, y_pred)
    print(Fore.LIGHTBLUE_EX + "Accuracy:", round(accuracy, 4))

    f1 = f1_score(y_true, y_pred)
    print(Fore.LIGHTBLUE_EX + "F1-score:", round(f1, 4))

    cm = confusion_matrix(y_true, y_pred)
    print(Fore.LIGHTBLUE_EX + "Confusion matrix:\n", cm)

    return accuracy, f1, cm

def truncated_svd(X_train, X_test, random_state, n_components=100):
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    svd.fit(X_train)
    transformed_X_train = svd.transform(X_train)
    transformed_X_test = svd.transform(X_test)
    return transformed_X_train, transformed_X_test

def logistic_regression(X_train, y_train, X_test, y_test, random_state, regularization="ridge", regularization_strength=1):
    """
    regularization_strength: inverse of alpha coefficient in regularization
    Possible regularizations: 'lasso', 'ridge', None
    """

    if regularization == "lasso":
        logreg = LogisticRegression(penalty="l1", solver="liblinear", C=regularization_strength, random_state=random_state)
    elif regularization == "ridge":
        logreg = LogisticRegression(penalty="l2", solver="lbfgs", C=regularization_strength, random_state=random_state)
    else:
        logreg = LogisticRegression(penalty=None, solver="lbfgs", C=regularization_strength, random_state=random_state)

    logreg.fit(X_train, y_train)

    # Making predictions on the test set
    y_pred = logreg.predict(X_test)

    # Evaluate
    accuracy, f1, cm = evaluate("Logistic Regression - Seed: " + str(random_state) + " - Regularization: " + str(regularization) + " - Regularization_strength: " + str(regularization_strength), y_test, y_pred)

    return accuracy, f1, cm

def support_vector_machine(X_train, y_train, X_test, y_test, random_state, misclass_penalty=1.0, kernel="rbf"):
    """
    misclass_penalty: penalty for misclassifying a data point, smaller ~ large margin
    Possible kernels: 'linear', 'poly' (degree 3), 'rbf', 'sigmoid'
    """
    
    svm = SVC(C=misclass_penalty, kernel=kernel, random_state=random_state)

    svm.fit(X_train, y_train)

    # Making predictions on the test set
    y_pred = svm.predict(X_test)

    # Evaluate
    accuracy, f1, cm = evaluate("Support Vector Machine - Seed: " + str(random_state) + " - Misclass_penalty: " + str(misclass_penalty) + " - Kernel: " + str(kernel), y_test, y_pred)

    return accuracy, f1, cm

def k_nearest_neighbors(X_train, y_train, X_test, y_test, n_neighbors=5, neighbor_weight="uniform", p=2, metric="minkowski"):
    """
    n_neighbors = number of neighbors
    Possible neighbor weights: 'uniform', 'distance', [callable]
    Possible metrics: 'cityblock' (manhattan), 'cosine', 'euclidean', 'minkowski' (p: power parameter)
    """
    
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=neighbor_weight, p=p, metric=metric)

    knn.fit(X_train, y_train)

    # Making predictions on the test set
    y_pred = knn.predict(X_test)

    # Evaluate
    if metric == "minkowski": 
        accuracy, f1, cm = evaluate("K Nearest Neighbors" + " - N_neighbors: " + str(n_neighbors) + " - Neighbor_weight: " + str(neighbor_weight) + " - p: " + str(p) + " - Metric: " + str(metric), y_test, y_pred)
    else:
        accuracy, f1, cm = evaluate("K Nearest Neighbors" + " - N_neighbors: " + str(n_neighbors) + " - Neighbor_weight: " + str(neighbor_weight) + " - Metric: " + str(metric), y_test, y_pred)

    return accuracy, f1, cm
