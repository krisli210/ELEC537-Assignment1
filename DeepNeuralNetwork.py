from threelayerneuralnetwork import *

class DeepNeuralNetwork(NeuralNetwork):

    def __init__(self, layers, seed=0):
        '''
        :param layers: the layers in sequential order of the neural network
        :param seed: random seed
        '''
        self.layers = layers
        self.input_dim = self.layers[0].n_inputs
        self.output_dim = self.layers[-1].n_nodes
        self.probs = None

    def actFun(self, z, type):
        '''
        actFun computes the activation functions
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: activations
        '''

        # YOU IMPLMENT YOUR actFun HERE
        match type:
            case 'tanh':
                out = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
            case 'sigmoid':
                out = 1 / (1 + np.exp(-z))
            case 'relu':
                out = np.maximum(0, z)
            case _:
                out = None
        return out

    def diff_actFun(self, z, type):
        '''
        diff_actFun computes the derivatives of the activation functions wrt the net input
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: the derivatives of the activation functions wrt the net input
        '''

        # YOU IMPLEMENT YOUR diff_actFun HERE
        match type:
            case 'tanh':
                out = 1 - ((np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))) ** 2
            case 'sigmoid':
                out = (1 / (1 + np.exp(-z))) * (1 - (1 / (1 + np.exp(-z))))
            case 'relu':
                out = np.where(z > 0, 1, 0)
            case _:
                out = None

        return out

    def feedforward(self, X):
        tensor = X
        for layer in self.layers:
            tensor = layer(tensor)
        self.probs = tensor
        return tensor

    def __call__(self, X):
        return self.feedforward(X)

    def backprop(self, y):
        delta_term = y
        for layer in reversed(self.layers):
            delta_term = layer.backprop(delta_term)

    def calculate_loss(self, X, y):
        '''
        calculate_loss computes the loss for prediction
        :param X: input data
        :param y: given labels - Shape (samples,)
        :return: the loss for prediction
        '''
        num_examples = len(X)

        self.feedforward(X)
        data_loss = np.sum(-np.log(self.probs[np.arange(num_examples), y]))
        return (1. / num_examples) * data_loss

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        '''
        fit_model uses backpropagation to train the network
        :param X: input data
        :param y: given labels
        :param num_passes: the number of times that the algorithm runs through the whole dataset
        :param print_loss: print the loss or not
        :return:
        '''
        # Gradient descent.
        for i in range(0, num_passes):
            self.feedforward(X)
            self.backprop(y)

            for layer in self.layers:
                layer.step(epsilon)

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" %
                      (i, self.calculate_loss(X, y)))

    def predict(self, X):
        '''
        predict infers the label of a given data point X
        :param X: input data
        :return: label inferred
        '''
        self.feedforward(X)
        return np.argmax(self.probs, axis=1)

class Layer(object):

    def __init__(self,n_inputs,n_nodes,actFun_type,reg_lambda = 0.01,seed=0,last_layer=False):
        self.n_inputs = n_inputs
        self.n_nodes = n_nodes
        self.actFun_type = actFun_type
        self.last_layer = last_layer
        self.reg_lambda = reg_lambda

        # Initialize weights
        np.random.seed(seed)
        self.W = np.random.randn(self.n_inputs, self.n_nodes) / np.sqrt(self.n_nodes)
        self.b = np.zeros((1, self.n_nodes))

        if last_layer == True:
            self.probs = None

    def actFun(self, z):
        '''
        actFun computes the activation functions
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: activations
        '''

        # YOU IMPLMENT YOUR actFun HERE
        match self.actFun_type:
            case 'tanh':
                out = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
            case 'sigmoid':
                out = 1 / (1 + np.exp(-z))
            case 'relu':
                out = np.maximum(0, z)
            case _:
                out = None
        return out

    def diff_actFun(self, z):
        '''
        diff_actFun computes the derivatives of the activation functions wrt the net input
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: the derivatives of the activation functions wrt the net input
        '''

        # YOU IMPLEMENT YOUR diff_actFun HERE
        match self.actFun_type:
            case 'tanh':
                out = 1 - ((np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))) ** 2
            case 'sigmoid':
                out = (1 / (1 + np.exp(-z))) * (1 - (1 / (1 + np.exp(-z))))
            case 'relu':
                out = np.where(z > 0, 1, 0)
            case _:
                out = None

        return out

    def feedforward(self, X):
        '''
        feedforward builds a 3-layer neural network and computes the two probabilities,
        one for class 0 and one for class 1
        :param X: layer input
        :param actFun: activation function
        :return:
        '''
        self.input = X
        self.z = X @ self.W + self.b
        if self.last_layer == True:
            # do softmax activation
            exp_scores = np.exp(self.z)
            self.a = exp_scores / (np.sum(exp_scores, axis=1, keepdims=True))
        else:
            self.a = self.actFun(self.z)
        return self.a

    def backprop(self,y):
        '''
        backprop implements backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: layer input
        :param y: layer output
        :return: dL/dW1, dL/b1, dL/dW2, dL/db2
        '''
        if self.last_layer == False:
            da = self.diff_actFun(self.z)
            delta = y * da
            self.dW = self.input.T @ delta
            self.db = np.sum(delta, axis=0)

        else: # need softmax here
            num_examples = len(self.input)
            delta = self.a
            delta[range(num_examples), y] -= 1

            dW_reg = (self.reg_lambda * self.W)
            self.dW = self.input.T @ delta + dW_reg
            self.db = np.sum(delta, axis=0)

        return delta @ self.W.T

    def __call__(self, x):
        return self.feedforward(x)

    def step(self, epsilon):
        self.W += -epsilon * self.dW
        self.b += -epsilon * self.db
