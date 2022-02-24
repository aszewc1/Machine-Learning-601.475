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
        #initiate variables
        w = self.w
        lr = self.lr
        X = X.toarray()
        ft_num = w.shape[0]
        
        #cycle through each row of x to update the features in w
        for i in range(X.shape[0]):
            curr_w = w
            sig = np.matmul(curr_w.transpose(), X[i][:].reshape((ft_num, 1)))
            w = (curr_w).reshape((ft_num, 1)) + lr * (X[i][:]).reshape((ft_num, 1)) * (y[i] - sigmoid(sig[0]))[0]
       
        self.w = w

    def predict(self, X):
        #get rid of sparse matrix type
        X = X.toarray()
        
        #slice w to fit the appropriate number of testing features
        w_slice = []
        w = self.w
        for i in range(X.shape[1]):
            w_slice.append(w[i])
        w_slice = np.array(w_slice)
       
        #multiply w and X to get probabilities
        res = sigmoid(np.matmul(X, w_slice))

        #perform classification predictions (0 or 1) based on probabilities
        for i in range(res.shape[0]):
            if res[i][0] < .5:
                res[i][0] = 0
            else:
                res[i][0] = 1
        
        #return final predictions
        return (res)

class LogisticRegressionNewton(Model):

    def __init__(self, n_features):
        super().__init__()
        # TODO: Initialize parameters
        self.w = np.zeros((n_features, 1))

    def fit(self, X, y):
        # TODO: Write code to fit the model
        pass

    def predict(self, X):
        # TODO: Write code to make predictions
        pass
