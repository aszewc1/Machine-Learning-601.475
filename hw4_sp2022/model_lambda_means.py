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
            
            r = [0] * len(mu)
            for n in range(X.shape[0]):
               
                #find closest existing cluster

                dist = np.linalg.norm(np.array(mu) - X[n][:], axis = 1)
                #I think problem is in line 99/100 --> trying to look at piazza 436 question to vectorize and it's going poorly
                if any(dist <= self.lambda0):
                    r[(dist.tolist()).index(min(dist.tolist()))] = 1
                
                #check for/create new cluster
                if min(dist) > self.lambda0:
                    r.append(1)
                    mu.append(X[n][:])
                else:
                    r.append(0)

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
                


