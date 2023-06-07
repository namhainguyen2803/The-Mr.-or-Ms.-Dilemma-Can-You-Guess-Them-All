import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import log_loss, precision_recall_curve, PrecisionRecallDisplay, roc_curve, RocCurveDisplay, roc_auc_score

class Metric:
    def __init__(self, y_true:np.ndarray, y_pred:np.ndarray, y_proba:np.ndarray=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba

    # Accuracy
    def getAccuracy(self):
        return accuracy_score(self.y_true, self.y_pred)
    
    # Recall
    def getRecallClass0(self):
        return recall_score(self.y_true, self.y_pred, pos_label=0, average="binary")
    def getRecallClass1(self):
        return recall_score(self.y_true, self.y_pred, pos_label=1, average="binary")
    def getMacroAvgRecall(self):
        return recall_score(self.y_true, self.y_pred, average="macro")
    def getWeightedAvgRecall(self):
        return recall_score(self.y_true, self.y_pred, average="weighted")

    # Precision
    def getPrecisionClass0(self):
        return precision_score(self.y_true, self.y_pred, pos_label=0, average="binary")
    def getPrecisionClass1(self):
        return precision_score(self.y_true, self.y_pred, pos_label=1, average="binary")
    def getMacroAvgPrecision(self):
        return precision_score(self.y_true, self.y_pred, average="macro")
    def getWeightedAvgPrecision(self):
        return precision_score(self.y_true, self.y_pred, average="weighted")
    
    # F1
    def getF1Class0(self):
        return f1_score(self.y_true, self.y_pred, pos_label=0, average="binary")
    def getF1Class1(self):
        return f1_score(self.y_true, self.y_pred, pos_label=1, average="binary")
    def getMacroAvgF1(self):
        return f1_score(self.y_true, self.y_pred, average="macro")
    def getWeightedAvgF1(self):
        return f1_score(self.y_true, self.y_pred, average="weighted")
    
    # Confusion matrix
    def getConfusionMatrix(self):
        cm = confusion_matrix(self.y_true, self.y_pred)
        cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
        return cm, cm_disp
    
    # Classification report
    def getClassificationReport(self):
        return classification_report(self.y_true, self.y_pred)
    
    ##########

    # Log loss
    def getLogLoss(self):
        return log_loss(self.y_true, self.y_proba)

    # Precision-recall curve
    def getPrecisionRecallCurveClass0(self):
        prec, recall, _ = precision_recall_curve(self.y_true, 1 - self.y_proba, pos_label=0)
        pr_disp = PrecisionRecallDisplay(prec, recall)
        return pr_disp
    
    def getPrecisionRecallCurveClass1(self):
        prec, recall, _ = precision_recall_curve(self.y_true, self.y_proba, pos_label=1)
        pr_disp = PrecisionRecallDisplay(prec, recall)
        return pr_disp

    # ROC curve
    def getRocCurveClass0(self):
        fpr, tpr, _ = roc_curve(self.y_true, 1 - self.y_proba, pos_label=0)
        roc_disp = RocCurveDisplay(fpr=fpr, tpr=tpr)
        return roc_disp
    
    def getRocCurveClass1(self):
        fpr, tpr, _ = roc_curve(self.y_true, self.y_proba, pos_label=1)
        roc_disp = RocCurveDisplay(fpr=fpr, tpr=tpr)
        return roc_disp
    
    def getRocAucScore(self):
        return roc_auc_score(self.y_true, self.y_proba)