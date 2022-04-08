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
        # initialize parameter
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
        numInstances = X.shape[0]

        # cluster initialization
        initial = np.sum(X, axis=0)/numInstances
        lambd = sum(np.linalg.norm(x - initial) for x in X)/numInstances
        lambd = lambd if self.lambda0 == 0 else self.lambda0
        mu = []
        mu.append(initial)

        # repeat for given number of iterations
        for i in range(iterations):

            # e step
            r = np.zeros((numInstances, len(mu)))
            for n in range(numInstances):

                example = X[n, :]

                # find closest existing cluster
                dist = np.linalg.norm(example - np.array(mu), axis=1)

                if any(dist <= lambd):
                    r[n, np.argmin(dist)] = 1
                else:
                    mu.append(example)
                    if len(mu) > r.shape[1]:
                        # expand array if too small
                        r = np.append(r, np.zeros(
                            (numInstances, len(mu))), axis=1)
                    r[n, len(mu)-1] = 1

            # m step
            for k in range(len(mu)):
                mu[k] = sum(X[r[:, k] == 1, :])/sum(r[:, k])

        self.mu = mu

    def predict(self, X):
        """ Predict.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        # initializations
        mu = self.mu
        X = X.toarray()
        numInstances = X.shape[0]
        labels = []

        # loop through examples in x
        for n in range(numInstances):

            example = X[n, :]

            best_dist = np.inf
            best_k = 0

            for k in range(len(mu)):

                # find distance between n and cluster
                curr_dist = np.linalg.norm(example - mu[k])

                # update closest cluster
                if curr_dist < best_dist:
                    best_dist = curr_dist
                    best_k = k

            labels.append = best_k

        return labels
