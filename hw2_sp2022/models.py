import numpy as np

def sigmoid(x):
    x = np.clip(x, a_min = -709, a_max = 709)
    return 1 / (1 + np.exp(-x))

class Model(object):

    def __init__(self):
        self.num_input_features = None

    def fit(self, X, y):
        """ Fit the model.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            y: A dense array of ints with shape [num_examples].
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

class LogisticRegressionSGD(Model):

    def __init__(self, n_features, learning_rate = 0.1):
        super().__init__()
        # TODO: Initialize parameters, learning rate
        self.w = np.zeros((n_features, 1))
        self.lr = learning_rate
        

    def fit(self, X, y):
        w = self.w
        lr = self.lr
        X = X.toarray()
        ft_num = w.shape[0]
        print(X.shape)
        for i in range(X.shape[0]):
            for j in range(w.shape[0]):
                sig = np.matmul(w.transpose(), X[i][:].reshape((ft_num, 1)))
                w[j] = w[j] + (lr * X[i][j]) * (y[i] - sigmoid(sig[0]))
        self.w = w

    def predict(self, X):
        # TODO: Write code to make predictions
        X = X.toarray()
        res = sigmoid(np.matmul(X, self.w))
        for i in range(res.shape[0]):
            if res[i][0] < .5:
                res[i][0] = 0
            else:
                res[i][0] = 1
        return (res)

class LogisticRegressionNewton(Model):

    def __init__(self, n_features):
        super().__init__()
        # TODO: Initialize parameters
        pass

    def fit(self, X, y):
        # TODO: Write code to fit the model
        pass

    def predict(self, X):
        # TODO: Write code to make predictions
        pass
