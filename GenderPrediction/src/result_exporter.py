from models import *
import os

def export_result(model:MyModel, random_state:int, drop_dup:bool, truncate:bool):
    accuracy = round((model.cm[0][0] + model.cm[1][1]) / (model.cm[0][0] + model.cm[1][1] + model.cm[1][0] + model.cm[0][1]), 4)
    
    parent_directory = os.path.abspath(os.path.join(os.getcwd(), ".."))
    file_path = os.path.join(parent_directory, "results/" + str(model.name) + "-drop_dup-" + str(drop_dup) + "-truncate-" + str(truncate) + "-random_state-" + str(random_state) + ".txt")
    
    with open(file_path, 'w') as f:
        f.write(str(model.name) + '\n')
        f.write("random-state = " + str(random_state) + "     drop_dup = " + str(drop_dup) + "     truncate = " + str(truncate) + '\n') 
        f.write('\n')
        f.write("CV based on " + str(model.scoring) + '\n')
        f.write("Best hyperparameters: " + str(model.best_params) + '\n')
        f.write("Best validation score: " + str(model.best_score) + '\n')
        f.write('\n')
        f.write("Metrics"+ '\n')
        f.write("Confusion matrix:" + '\n')
        f.write(str(model.cm) + '\n')
        f.write("Accuracy: " + str(accuracy) + '\n')
        f.write("Classification report:" + '\n')
        f.write(str(model.classification_report) + '\n')
        f.write("Log loss: " + str(model.log_loss) + '\n')
        f.write("ROC-AUC score: " + str(model.roc_auc) + '\n')
        f.write("Training time taken: " + str(model.time_taken) + '\n')
