from sklearn.model_selection import GridSearchCV
from metrics import Metric

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Visual purpose
import warnings
import colorama
from colorama import Fore
colorama.init(autoreset=True)

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

    def fit(self):
        if self.grid_search == False:
            self.model.fit(self.X_train, self.y_train)
        elif self.grid_search == True:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.gs.fit(self.X_train, self.y_train)
                
                print(Fore.LIGHTYELLOW_EX + "CV based on " + self.scoring)
                print(Fore.LIGHTBLUE_EX + "Best hyperparameters:", self.gs.best_params_)
                print(Fore.LIGHTBLUE_EX + "Best validation score:",round(self.gs.best_score_, 4))

    def predict(self):
        if self.grid_search == False:
            return self.model.predict(self.X_test)
        elif self.grid_search == True:
            return self.gs.best_estimator_.predict(self.X_test)

    def predict_proba(self):
        if self.grid_search == False:
            return self.model.predict_proba(self.X_test)
        elif self.grid_search == True:
            return self.gs.best_estimator_.predict_proba(self.X_test)
    
    def evaluate(self):
        print("------------------------------------------------------------")
        self.fit()
        print(Fore.LIGHTYELLOW_EX + "Metrics")
        y_pred = self.predict()
        y_proba = self.predict_proba()
        
        self.metric = Metric(self.y_test, y_pred, y_proba[:, 1])

        cm, _ = self.metric.getConfusionMatrix()
        print(Fore.LIGHTBLUE_EX + "Confusion matrix:")
        print(cm)

        classification_report = self.metric.getClassificationReport()
        print(Fore.LIGHTBLUE_EX + "Classification report:")
        print(classification_report)

        log_loss = self.metric.getLogLoss()
        print(Fore.LIGHTBLUE_EX + "Log loss:")
        print(round(log_loss, 4))

        roc_auc = self.metric.getRocAucScore()
        print(Fore.LIGHTBLUE_EX + "ROC-AUC score:")
        print(round(roc_auc, 4))
        print("------------------------------------------------------------")

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
            
    def fit(self):
        print(Fore.LIGHTYELLOW_EX + "LOGISTIC REGRESSION")
        super().fit()

class MySupportVectorMachine(MyModel):
    """
    misclass_penalty: penalty for misclassifying a data point, smaller ~ large margin
    Possible kernels: 'linear', 'poly' (degree 3), 'rbf', 'sigmoid'
    """
    def __init__(self, X_train, X_test, y_train, y_test, random_state,
                 misclass_penalty=1.0, kernel="rbf",
                 grid_search=False, scoring=None, param_grid=None, cv=0):
        
        super().__init__(X_train, X_test, y_train, y_test, grid_search, scoring)

        if grid_search == False:
            self.model = SVC(C=misclass_penalty, kernel=kernel, random_state=random_state)
        
        elif grid_search == True:
            self.model = SVC(random_state=random_state)
            self.gs = GridSearchCV(estimator=self.model, scoring=scoring, param_grid=param_grid, cv=cv)
    
    def fit(self):
        print(Fore.LIGHTYELLOW_EX + "SUPPORT VECTOR MACHINE")
        super().fit()

class MyKNearestNeighbor(MyModel):
    """
    n_neighbors = number of neighbors
    Possible neighbor weights: 'uniform', 'distance', [callable]
    Possible metrics: 'cityblock' (manhattan), 'cosine', 'euclidean', 'minkowski' (p: power parameter)
    """
    def __init__(self, X_train, X_test, y_train, y_test,
                 n_neighbors=5, neighbor_weight="uniform", p=2, metric="minkowski",
                 grid_search=False, scoring=None, param_grid=None, cv=0):
        
        super().__init__(X_train, X_test, y_train, y_test, grid_search, scoring)

        if grid_search == False:
            self.model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=neighbor_weight, p=p, metric=metric)
        
        elif grid_search == True:
            self.model = KNeighborsClassifier()
            self.gs = GridSearchCV(estimator=self.model, scoring=scoring, param_grid=param_grid, cv=cv)
    
    def fit(self):
        print(Fore.LIGHTYELLOW_EX + "K NEAREST NEIGHBORS")
        super().fit()