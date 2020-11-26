import os
import numpy as np
from matplotlib import pyplot
from scipy import optimize
from scipy.io import loadmat
import utils


class NeuralNetwork:
    '''
        1 hidden layered neural network
    '''
    def __init__(self):
        #  training data stored in arrays X, y
        data = loadmat(os.path.join('Data', 'ex4data1.mat'))
        self.X, self.y = data['X'], data['y'].ravel()

        # set the zero digit to 0, rather than its mapped 10 in this dataset
        # This is an artifact due to the fact that this dataset was used in
        # MATLAB where there is no index 0
        self.y[self.y == 10] = 0

        # Setup the parameters you will use for this exercise
        self.input_layer_size = 400  # 20x20 Input Images of Digits
        self.hidden_layer_size = 25   # 25 hidden units
        self.num_labels = 10          # 10 labels, from 0 to 9

        # Load the weights into variables Theta1 and Theta2
        self.weights = loadmat(os.path.join('Data', 'ex4weights.mat'))

        # Theta1 has size 25 x 401
        # Theta2 has size 10 x 26
        self.Theta1, self.Theta2 = self.weights['Theta1'], self.weights['Theta2']

        # swap first and last columns of Theta2, due to legacy from MATLAB indexing,
        # since the weight file ex3weights.mat was saved based on MATLAB indexing
        self.Theta2 = np.roll(self.Theta2, 1, axis=0)

        # Unroll parameters
        self.nn_params = np.concatenate([self.Theta1.ravel(), self.Theta2.ravel()])

    def display_data(self):
        # Randomly select 100 data points to display
        m = self.y.shape[0]
        rand_indices = np.random.choice(m, 100, replace=False)
        sel = self.X[rand_indices, :]

        utils.displayData(sel)

    def nnCostFunction(self, nn_params,
                       input_layer_size,
                       hidden_layer_size,
                       num_labels,
                       X, y, lambda_=0.0):
        """
        Implements the neural network cost function and gradient for a two layer neural 
        network which performs classification. 
        
        Parameters
        ----------
        nn_params : array_like
            The parameters for the neural network which are "unrolled" into 
            a vector. This needs to be converted back into the weight matrices Theta1
            and Theta2.
        
        input_layer_size : int
            Number of features for the input layer. 
        
        hidden_layer_size : int
            Number of hidden units in the second layer.
        
        num_labels : int
            Total number of labels, or equivalently number of units in output layer. 
        
        X : array_like
            Input dataset. A matrix of shape (m x input_layer_size).
        
        y : array_like
            Dataset labels. A vector of shape (m,).
        
        lambda_ : float, optional
            Regularization parameter.
    
        Returns
        -------
        J : float
            The computed value for the cost function at the current weight values.
        
        grad : array_like
            An "unrolled" vector of the partial derivatives of the concatenatation of
            neural network weights Theta1 and Theta2.
        """
        # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
        # for our 2 layer neural network
        Theta1 = np.reshape(self.nn_params[:hidden_layer_size * (input_layer_size + 1)],
                            (hidden_layer_size, (input_layer_size + 1)))

        Theta2 = np.reshape(self.nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                            (num_labels, (hidden_layer_size + 1)))

        # Setup some useful variables
        m = y.size

        # You need to return the following variables correctly
        J = 0
        Theta1_grad = np.zeros(Theta1.shape)
        Theta2_grad = np.zeros(Theta2.shape)

        a1 = np.concatenate([np.ones((m, 1)), X], axis=1)

        a2 = utils.sigmoid(a1.dot(Theta1.T))
        a2 = np.concatenate([np.ones((a2.shape[0], 1)), a2], axis=1)

        a3 = utils.sigmoid(a2.dot(Theta2.T))

        y_matrix = y.reshape(-1)
        y_matrix = np.eye(num_labels)[y_matrix]

        temp1 = Theta1
        temp2 = Theta2

        # Add regularization term

        reg_term = (lambda_ / (2 * m)) * \
            (np.sum(np.square(temp1[:, 1:])) + np.sum(np.square(temp2[:, 1:])))

        J = (-1 / m) * np.sum((np.log(a3) * y_matrix) +
                            np.log(1 - a3) * (1 - y_matrix)) + reg_term

        # Backpropogation

        delta_3 = a3 - y_matrix
        delta_2 = delta_3.dot(Theta2)[:, 1:] * self.sigmoidGradient(a1.dot(Theta1.T))

        Delta1 = delta_2.T.dot(a1)
        Delta2 = delta_3.T.dot(a2)

        # Add regularization to gradient

        Theta1_grad = (1 / m) * Delta1
        Theta1_grad[:, 1:] = Theta1_grad[:, 1:] + (lambda_ / m) * Theta1[:, 1:]

        Theta2_grad = (1 / m) * Delta2
        Theta2_grad[:, 1:] = Theta2_grad[:, 1:] + (lambda_ / m) * Theta2[:, 1:]

        grad = np.concatenate([Theta1_grad.ravel(), Theta2_grad.ravel()])

        return J, grad

    def test_nnCostFunction(self):
        lambda_ = 0
        J, _ = self.nnCostFunction(self.nn_params, self.input_layer_size, self.hidden_layer_size,
                                   self.num_labels, self.X, self.y, lambda_)
        print('Cost at parameters (loaded from ex4weights): %.6f ' % J)
        print('The cost should be about                   : 0.287629.')

    def test_nnCostFunction_Regularized(self):
        # Weight regularization parameter (we set this to 1 here).
        lambda_ = 1
        J, _ = self.nnCostFunction(self.nn_params, self.input_layer_size, self.hidden_layer_size,
                                   self.num_labels, self.X, self.y, lambda_)

        print('Cost at parameters (loaded from ex4weights): %.6f' % J)
        print('This value should be about                 : 0.383770.')

    def sigmoidGradient(self, z):
        """
        Computes the gradient of the sigmoid function evaluated at z. 
        This should work regardless if z is a matrix or a vector. 
        In particular, if z is a vector or matrix, you should return
        the gradient for each element.
        
        Parameters
        ----------
        z : array_like
            A vector or matrix as input to the sigmoid function. 
        
        Returns
        --------
        g : array_like
            Gradient of the sigmoid function. Has the same shape as z. 
        """

        g = np.zeros(z.shape)

        # ====================== YOUR CODE HERE ======================
        g = utils.sigmoid(z) * (1 - utils.sigmoid(z))
        # =============================================================
        return g

    def test_sigmoid_gradient(self):
        z = np.array([-1, -0.5, 0, 0.5, 1])
        g = self.sigmoidGradient(z)
        print('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:\n  ')
        print(g)

    def randInitializeWeights(self, L_in, L_out, epsilon_init=0.12):
        """
        Randomly initialize the weights of a layer in a neural network.
        
        Parameters
        ----------
        L_in : int
            Number of incomming connections.
        
        L_out : int
            Number of outgoing connections. 
        
        epsilon_init : float, optional
            Range of values which the weight can take from a uniform 
            distribution.
        
        Returns
        -------
        W : array_like
            The weight initialiatized to random values.  Note that W should
            be set to a matrix of size(L_out, 1 + L_in) as
            the first column of W handles the "bias" terms.
        """

        # You need to return the following variables correctly
        W = np.zeros((L_out, 1 + L_in))

        # ====================== YOUR CODE HERE ======================
        W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init
        # ============================================================
        return W

    def test_nn_gradient(self):
        #  Check gradients by running checkNNGradients
        #  compares analytical method value with backprop values 
        lambda_ = 3
        utils.checkNNGradients(self.nnCostFunction, lambda_)

        # Also output the costFunction debugging values
        debug_J, _ = self.nnCostFunction(self.nn_params, self.input_layer_size,
                                    self.hidden_layer_size, self.num_labels, self.X, self.y, lambda_)

        print('\n\nCost at (fixed) debugging parameters (w/ lambda = %f): %f ' %
              (lambda_, debug_J))
        print('(for lambda = 3, this value should be about 0.576051)')

    def train(self):
        print('Initializing Neural Network Parameters ...')

        initial_Theta1 = self.randInitializeWeights(
            self.input_layer_size, self.hidden_layer_size)
        initial_Theta2 = self.randInitializeWeights(self.hidden_layer_size, self.num_labels)

        # Unroll parameters
        initial_nn_params = np.concatenate(
            [initial_Theta1.ravel(), initial_Theta2.ravel()], axis=0)
        #  After you have completed the assignment, change the maxiter to a larger
        #  value to see how more training helps.
        options = {'maxiter': 100}

        #  You should also try different values of lambda
        lambda_ = 1

        # Create "short hand" for the cost function to be minimized
        def costFunction(p): return self.nnCostFunction(p, self.input_layer_size,
                                                   self.hidden_layer_size,
                                                   self.num_labels, self.X, self.y, lambda_)

        # Now, costFunction is a function that takes in only one argument
        # (the neural network parameters)
        res = optimize.minimize(costFunction,
                                initial_nn_params,
                                jac=True,
                                method='TNC',
                                options=options)

        # get the solution of the optimization
        nn_params = res.x

        # Obtain Theta1 and Theta2 back from nn_params
        self.Theta1 = np.reshape(nn_params[:self.hidden_layer_size * (self.input_layer_size + 1)],
                            (self.hidden_layer_size, (self.input_layer_size + 1)))

        self.Theta2 = np.reshape(nn_params[(self.hidden_layer_size * (self.input_layer_size + 1)):],
                            (self.num_labels, (self.hidden_layer_size + 1)))

        def print_accuracy(self):
            self.predict(self.X)
            print('Training Set Accuracy: %f' % (np.mean(pred == y) * 100))

        def predict(X):
            return utils.predict(self.Theta1, self.Theta2, X)