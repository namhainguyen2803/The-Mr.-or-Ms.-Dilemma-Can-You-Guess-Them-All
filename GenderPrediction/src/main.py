# Basic imports
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Custom imports
import utils
import tf_idf

def main():
    # Set random seed
    RANDOM_STATE = 2023
    
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
    tfidf_X_train, tfidf_X_test = utils.truncated_svd(tfidf_X_train, tfidf_X_test, random_state=RANDOM_STATE)
    
    # Apply different models
    """
    utils.logistic_regression(tfidf_X_train, y_train, tfidf_X_test, y_test, 
                              random_state=RANDOM_STATE, 
                              regularization="ridge", 
                              regularization_strength=1)
    """
    """
    utils.support_vector_machine(tfidf_X_train, y_train, tfidf_X_test, y_test, 
                                 random_state=RANDOM_STATE, 
                                 misclass_penalty=5.0,
                                 kernel="linear",
                                 )
    """
    utils.k_nearest_neighbors(tfidf_X_train, y_train, tfidf_X_test, y_test,
                              n_neighbors=15,
                              neighbor_weight="distance",
                              p=2,
                              metric="cosine")

if __name__ == "__main__":
    main()