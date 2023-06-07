from sklearn.decomposition import TruncatedSVD

def truncated_svd(X_train, X_test, random_state, n_components=100):
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    svd.fit(X_train)
    transformed_X_train = svd.transform(X_train)
    transformed_X_test = svd.transform(X_test)
    return transformed_X_train, transformed_X_test 