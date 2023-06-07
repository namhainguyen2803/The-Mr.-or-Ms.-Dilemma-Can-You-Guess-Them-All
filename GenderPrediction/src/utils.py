from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GridSearchCV
from metrics import Metric

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Visual purpose
import colorama
from colorama import Fore
colorama.init(autoreset=True)
import warnings

def truncated_svd(X_train, X_test, random_state, n_components=100):
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    svd.fit(X_train)
    transformed_X_train = svd.transform(X_train)
    transformed_X_test = svd.transform(X_test)
    return transformed_X_train, transformed_X_test 

def support_vector_machine(X_train, y_train, X_test, y_test, random_state, 
                           misclass_penalty=1.0, kernel="rbf",
                           grid_search=False, scoring=None, param_grid=None, cv=0):
    """
    misclass_penalty: penalty for misclassifying a data point, smaller ~ large margin
    Possible kernels: 'linear', 'poly' (degree 3), 'rbf', 'sigmoid'
    """
    
    if grid_search == False:
        svm = SVC(C=misclass_penalty, kernel=kernel, random_state=random_state)

        svm.fit(X_train, y_train)

        # Making predictions on the test set
        y_pred = svm.predict(X_test)
    
    elif grid_search == True:
        svm = SVC(random_state=random_state)
        gs = GridSearchCV(estimator=svm, scoring=scoring, param_grid=param_grid, cv=cv)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            gs.fit(X_train, y_train)
        
        print(Fore.LIGHTYELLOW_EX + "Support Vector Machine - CV based on " + scoring)
        print(Fore.LIGHTBLUE_EX + "Best hyperparameters:", gs.best_params_)
        print(Fore.LIGHTBLUE_EX + "Best validation score:",round(gs.best_score_, 4))
        
        # Making predictions on the test set, WITH THE BEST ESTIMATOR OBTAINED
        y_pred = gs.best_estimator_.predict(X_test)

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

class MyModel:
    def __init__(self, X_train, X_test, y_train, y_test, grid_search, scoring):
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        self.grid_search = grid_search
        self.scoring = scoring
        
        self.model = None
        self.gs = None

    def fit(self, X_train, y_train):
        if self.grid_search == False:
            self.model.fit(X_train, y_train)
        elif self.grid_search == True:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.gs.fit(X_train, y_train)
                
                print(Fore.LIGHTYELLOW_EX + "CV based on " + self.scoring)
                print(Fore.LIGHTBLUE_EX + "Best hyperparameters:", self.gs.best_params_)
                print(Fore.LIGHTBLUE_EX + "Best validation score:",round(self.gs.best_score_, 4))

    def predict(self, X_test):
        if self.grid_search == False:
            return self.model.predict(X_test)
        elif self.grid_search == True:
            return self.gs.best_estimator_.predict(X_test)

    def predict_proba(self, X_test):
        if self.grid_search == False:
            return self.model.predict_proba(X_test)
        elif self.grid_search == True:
            return self.gs.best_estimator_.predict_proba(X_test)
    
    def evaluate(self):
        self.fit(self.X_train, self.y_train)
        print(Fore.LIGHTYELLOW_EX + "Metrics")
        y_pred = self.predict(self.X_test)
        y_proba = self.predict_proba(self.X_test)
        
        self.metric = Metric(self.y_test, y_pred, y_proba[:, 1])

        classification_report = self.metric.getClassificationReport()
        print(Fore.LIGHTBLUE_EX + "Classification report:")
        print(classification_report)

        log_loss = self.metric.getLogLoss()
        print(Fore.LIGHTBLUE_EX + "Log loss:")
        print(round(log_loss, 4))

class MyLogisticRegression(MyModel):
    """
    regularization_strength: inverse of alpha coefficient in regularization
    Possible regularizations: 'lasso', 'ridge', None
    """
    def __init__(self, X_train, X_test, y_train, y_test, random_state, 
                regularization="ridge", regularization_strength=1,
                grid_search=False, scoring=None, param_grid=None, cv=0):
        
        super().__init__(X_train, X_test, y_train, y_test, grid_search, scoring)

        if grid_search == False:
            if regularization == "lasso":
                self.model = LogisticRegression(penalty="l1", solver="liblinear", C=regularization_strength, random_state=random_state)
            elif regularization == "ridge":
                self.model = LogisticRegression(penalty="l2", solver="lbfgs", C=regularization_strength, random_state=random_state)
            else:
                self.model = LogisticRegression(penalty=None, solver="lbfgs", C=regularization_strength, random_state=random_state)
        
        elif grid_search == True:
            self.model = LogisticRegression(random_state=random_state)
            self.gs = GridSearchCV(estimator=self.model, scoring=scoring, param_grid=param_grid, cv=cv)
            
    def fit(self, X_train, y_train):
        print(Fore.LIGHTYELLOW_EX + "LOGISTIC REGRESSION")
        super().fit(X_train, y_train)