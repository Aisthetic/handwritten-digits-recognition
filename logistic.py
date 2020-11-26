import os
import numpy as np
from matplotlib import pyplot
from scipy import optimize
from scipy.io import loadmat
import utils


class LogisticRegression:
    def __init__(self):
        # 20x20 Input Images of Digits
        self.input_layer_size  = 400

        # 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)
        self.num_labels = 10

    def load_data(self):
        #  training data stored in arrays X, y
        self.data = loadmat(os.path.join('Data', 'ex3data1.mat'))
        self.X, self.y = self.data['X'], self.data['y'].ravel()

        # set the zero digit to 0, rather than its mapped 10 in this dataset
        # This is an artifact due to the fact that this dataset was used in 
        # MATLAB where there is no index 0
        self.y[self.y == 10] = 0

    def display_data(self):
        # Randomly select 100 data points to display
        rand_indices = np.random.choice(m, 100, replace=False)
        sel = X[rand_indices, :]

        utils.displayData(sel)


    def lrCostFunction(self, theta, X, y, lambda_):
        #Initialize some useful values
        m = y.size
        
        # convert labels to ints if their type is bool
        if y.dtype == bool:
            y = y.astype(int)
        
        J = 0
        grad = np.zeros(theta.shape)

        sigmoid = utils.sigmoid

        log = np.log
        h = sigmoid(np.dot(X, theta))
        biasless_theta = theta.copy()
        biasless_theta[0] = 0

        grad = (1/m) * np.dot( np.transpose(X), (h - y) ) + (lambda_/m)*biasless_theta

        J =  (-1/m) *  np.sum(( y * log(h) + (1-y) * log(1-h) )) + (lambda_/(2*m))*np.sum(biasless_theta**2)
            
        # =============================================================
        return J, grad

    def testlrCostFunction(self) : 
         # test values for the parameters theta
        theta_t = np.array([-2, -1, 1, 2], dtype=float)

        # test values for the inputs
        X_t = np.concatenate([np.ones((5, 1)), np.arange(1, 16).reshape(5, 3, order='F')/10.0], axis=1)

        # test values for the labels
        y_t = np.array([1, 0, 1, 0, 1])

        # test value for the regularization parameter
        lambda_t = 3
        J, grad = lrCostFunction(theta_t, X_t, y_t, lambda_t)

        print('Cost         : {:.6f}'.format(J))
        print('Expected cost: 2.534819')
        print('-----------------------')
        print('Gradients:')
        print(' [{:.6f}, {:.6f}, {:.6f}, {:.6f}]'.format(*grad))
        print('Expected gradients:')
        print(' [0.146561, -0.548558, 0.724722, 1.398003]');

    def oneVsAll(self , X, y, num_labels, lambda_):
        """
        Trains num_labels logistic regression classifiers and returns
        each of these classifiers in a matrix all_theta, where the i-th
        row of all_theta corresponds to the classifier for label i.
        
        Parameters
        ----------
        X : array_like
            The input dataset of shape (m x n). m is the number of 
            data points, and n is the number of features. Note that we 
            do not assume that the intercept term (or bias) is in X, however
            we provide the code below to add the bias term to X. 
        
        y : array_like
            The data labels. A vector of shape (m, ).
        
        num_labels : int
            Number of possible labels.
        
        lambda_ : float
            The logistic regularization parameter.
        
        Returns
        -------
        all_theta : array_like
            The trained parameters for logistic regression for each class.
            This is a matrix of shape (K x n+1) where K is number of classes
            (ie. `numlabels`) and n is number of features without the bias.
        """
        # Some useful variables
        m, n = self.X.shape
        
        # You need to return the following variables correctly 
        all_theta = np.zeros((num_labels, n + 1))

        # Add ones to the X data matrix
        X = np.concatenate([np.ones((m, 1)), X], axis=1)

        # ====================== YOUR CODE HERE ======================
        for c in range(num_labels):
            initial_theta = np.zeros(n + 1)
            options = {'maxiter': 50}
            res = optimize.minimize(lrCostFunction, 
                                    initial_theta, 
                                    (X, (y == c), lambda_), 
                                    jac=True, 
                                    method='TNC',
                                    options=options) 
            all_theta[c] = res.x

        # ============================================================
        return all_theta


    def predictOneVsAll(all_theta, X):
        """
        Return a vector of predictions for each example in the matrix X. 
        Note that X contains the examples in rows. all_theta is a matrix where
        the i-th row is a trained logistic regression theta vector for the 
        i-th class. You should set p to a vector of values from 0..K-1 
        (e.g., p = [0, 2, 0, 1] predicts classes 0, 2, 0, 1 for 4 examples) .
        
        Parameters
        ----------
        all_theta : array_like
            The trained parameters for logistic regression for each class.
            This is a matrix of shape (K x n+1) where K is number of classes
            and n is number of features without the bias.
        
        X : array_like
            Data points to predict their labels. This is a matrix of shape 
            (m x n) where m is number of data points to predict, and n is number 
            of features without the bias term. Note we add the bias term for X in 
            this function. 
        
        Returns
        -------
        p : array_like
            The predictions for each data point in X. This is a vector of shape (m, ).
        """
        m = X.shape[0]
        num_labels = all_theta.shape[0]

        # You need to return the following variables correctly 
        p = np.zeros(m)
        # Add ones to the X data matrix
        X = np.concatenate([np.ones((m, 1)), X], axis=1)

        # ====================== YOUR CODE HERE ======================
        h = utils.sigmoid(np.dot(all_theta,np.transpose(X)))

        p = np.argmax(h, axis=0)
        
        # ============================================================
        return p

    def testOneVsAll(self):
        lambda_ = 0.1
        all_theta = oneVsAll(X, y, num_labels, lambda_)
        pred = predictOneVsAll(all_theta, X)
        print('Training Set Accuracy: {:.2f}%'.format(np.mean(pred == y) * 100))