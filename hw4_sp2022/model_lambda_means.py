""" 
Keep model implementations in here.

This file is where you will write all of your code!
"""

import numpy as np
from tqdm import tqdm

class Model(object):
    """ Abstract model object.

    Contains a helper function which can help with some of our datasets.
    """

    def __init__(self, nfeatures):
        self.num_input_features = nfeatures


    def fit(self, *, X, iterations):
        """ Fit the model.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            iterations: int giving number of clustering iterations
        """
        raise NotImplementedError()


    def predict(self, X):
        """ Predict.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()


    def _fix_test_feats(self, X):
        """ Fixes some feature disparities between datasets.
        Call this before you perform inference to make sure your X features
        match your weights.
        """
        num_examples, num_input_features = X.shape
        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)
        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]
        return X


class LambdaMeans(Model):

    def __init__(self, *, nfeatures, lambda0):
        super().__init__(nfeatures)
        """
        Args:
            nfeatures: size of feature space (only needed for _fix_test_feats)
            lambda0: A float giving the default value for lambda
        """
        #initialize parameters
        self.nfeatures = nfeatures 
        self.lambda0 = lambda0

    def fit(self, *, X, iterations):
        """
        Fit the LambdaMeans model.
        Note: labels are not used here.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            iterations: int giving number of clustering iterations
        """
        # TODO: Implement this!
        
        X = X.toarray()
        print(X)
        #cluster initialization
        mu = [sum(X)/len(X)]
        clusters = 1
        #repeat for given number of iterations
        for i in range(iterations):
            #e step
            for k in range(clusters):
                r = 0
                if k == np.linalg.norm(X[i] - mu[i]) and np.linalg.norm(X[i] - mu) <= self.lambda0:
                    r = 1
                r_new_clust = 0
                if np.linalg.norm(X[i] - mu[i]) > self.lambda0:
                    r_new_clust = 1
                if r_new_clust == 1:
                    mu[i+1] = X[i]
            #mstep
            for k in range(clusters):
                mu[k] = sum(r*X[i])/sum(r)

    def predict(self, X):
        """ Predict.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        # TODO: Implement this!
        raise Exception("You must implement this method!")
