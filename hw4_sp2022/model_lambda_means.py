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
        X = X.toarray()
        #cluster initialization
        initial = np.sum(X, axis=0)/len(X)
        mu = []
        mu.append(initial)
        
        #repeat for given number of iterations
        for i in range(iterations):

            #e step
            r_tot = []
            
            for n in range(X.shape[0]):
                for k in range(len(mu)):
                    r = 0
                    if k == np.linalg.norm(X[n][:] - mu[k]) and np.linalg.norm(X[n][:] - mu[k]) <= self.lambda0:
                        r = 1
                    r_tot.append(r)

                    r_new_clust = 0
                    if np.linalg.norm(X[i] - mu[k]) > self.lambda0:
                        r_new_clust = 1
                    if r_new_clust == 1:
                        mu.append(X[n][:])
            #m step
            for k in range(len(mu)):
                mu[k] = (sum(r)*np.sum(X, axis = 0))/sum(r)
        self.mu = mu

    def predict(self, X):
        """ Predict.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        #initializations
        mu = self.mu
        X = X.toarray()
        labels = list(range(0, X.shape[0]))
        #loop through examples in x
        for n in range(X.shape[0]):
            best_dist = 100000000000
            best_k = 0
            for k in range(len(mu)):
                #find distance between n and cluster
                curr_dist = np.linalg.norm(X[n][:] - mu[k])
                #update closest cluster
                if curr_dist < best_dist:
                    curr_dist = best_dist
                    best_k = k
                #tie breaking
                elif curr_dist == best_dist:
                    if k < best_k:
                        curr_dist = best_dist
                        best_k = k
            labels[n] = best_k
        return labels
                


