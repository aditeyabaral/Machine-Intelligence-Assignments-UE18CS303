'''
ARCHITECTURE AND HYPERPARAMERTERS

1.  The neural network designed in the following code below is a simple feed-forward neural network architecture.

2.  The network has 3 fully connected dense layers, with 128, 32 and 1 neuron respectively.

3.  The first and the second layer use a ReLu activation function while the last layer uses a Sigmoid function.

4.  The cost function used is the binary cross-entropy loss function, defined as:
    J(y, y_hat) = -(y log(y_hat) + (1-y)log(1-y_hat))

5.  The layers also use Xavier's initialisation metric (as presented in the paper Understanding the difficulty 
    of training deep feedforward neural networks. In Aistats, volume 9, pages 249–256, 2010) for the initial values
    of the weights and the bias is initialised to small values from 0-1.

6.  We have also utitlised L2 regularization and Adam Optimisation (as presented in the paper
    Adam: A method for Stochastic Optimization, ICLR 2015) as the regularizer and optimizer respectively.

7.  The architecture defined uses the recursive backpropagation approach/equation, and hence can adapt to any change in
    the number of layers as well as the neurons per layer - as long as the first L-1 layers use a ReLu activation function
    and the Lth layer uses a Sigmoid activation function.

8.  It accepts the following HYPERPARAMETERS - 
        1. The model initialiser takes in the number of layers as a list containing the neurons per layer.
       For example, if we use 2 layers with 16 and 2 neurons, we create the model with: 
       model = NN([16, 2])

        2. The fit() function takes the following hyperparameters
            2.1 learning_rate: The learning rate to be used for Gradient Descent with a value set at 0.0001
       
            2.2 epochs: Iterations to train the model for, with a value set at 1000
       
            2.3 regularize: Boolean values to indicate whether L2 regularization has been employed
       
            2.4 l: Value of regularization parameter "lambda" 
       
            2.5 beta1: The weight associated with the moving average of the first moment
                  of the Adam Optimisation algorithm, set at 0.9
       
            2.6 beta2: The weight associated with the moving average of the second moment
                  of the Adam Optimisation algorithm, set at 0.999
       
            2.7 epsilon: Factor for numerical stability in division to prevent 
                    Division by Zero in the Adam Optimisation algorithm, set at 10^-8

            2.8 initialize: Boolean values to indicate reset of weights and bias

KEY FEATURES OF IMPLEMENTATION AND ADVANCED FEATURES

1.  The model works for an arbitrary number of layers and is capable of performing forward propagation as well as
    backpropagation for any given combination of layers and neurons per layer. This is owing to the fact that the 
    implementation uses the recursive backpropagation equation to store the partial derivatives of each successive layer
    and use these derivatives in the earlier layers: 
        dJ/dW[i] = dJ/dZ[i] * A[i-1]'
        dJ/db[i] =  SUM {dJ/dZ[i]}

2.  The model also utilises the Xavier method of initilisation of weights so that the values are sampled from a normal
    distribution and uses the Adam optimiser to speed up training. 

3. The model finally uses L2 regularization to prevent overfitting, since the dataset is imbalanced.

'''


# importing neccessary libraries
import numpy as np  # for matrix computations
import pandas as pd  # to read the dataframe
# to split into train and test sets
from sklearn.model_selection import train_test_split

np.random.seed(0)   # set a random seed to obtain consistent results


def relu(x):
    '''
    Obtain the ReLu activated matrix of x
    '''
    return np.maximum(x, 0)  # ReLu(x) = max(x, 0)


def gradRelu(x):
    '''
    Obtain the gradient of ReLu wrt x
    '''
    return np.where(x <= 0, 0, 1)   # ReLu'(x) = {1 if x>0 else 0}


def sigmoid(x):
    '''
    Obtain the sigmoid activated matrix of x
    '''
    return 1/(1+np.exp(-x))  # sigmoid(x) = 1/(1+e^-x)


def gradSigmoid(x):
    '''
    Obtain the gradient of sigmoid wrt x
    '''
    s = sigmoid(x)
    return s*(1-s)  # sigmoid'(x) = sigmoid(x)(1-sigmoid(x))


def getLayerWeight(neurons_current, neurons_previous, mode="small", mean=0, stddev=1):
    '''
    Obtain the weights associated with a layer. 
    Size of the matrix returned is (neurons_current, neurons_previous)
    The mode of initialisation can be zeros, small random numbers, Xavier or He-et-al
    '''
    if mode == "zero":
        w = np.zeros((neurons_current, neurons_previous))
    elif mode == "normal":
        w = stddev * np.random.randn(neurons_current, neurons_previous) + mean
    elif mode == "xavier":
        w = np.random.randn(neurons_current, neurons_previous) * \
            np.sqrt(2/(neurons_current + neurons_previous))
    elif mode == "he":
        w = np.random.randn(neurons_current, neurons_previous) * \
            np.sqrt(1/neurons_previous)
    else:
        w = np.random.random((neurons_current, neurons_previous))

    return w


def getLayerBias(neurons, mode="zero", mean=0, stddev=1):
    '''
    Obtain the bias associated with a layer. 
    Size of the matrix returned is (neurons, 1)
    The mode of initialisation can be zeros or small random numbers
    '''
    if mode == "small":
        b = np.random.random((neurons, 1))
    elif mode == "normal":
        b = stddev * np.random.randn(neurons, 1) + mean
    else:
        b = np.zeros((neurons, 1))

    return b


def activate(x, function):
    '''Activate a matrix of inputs using the activation function provided. 
    Accepted activation functions are "relu" and "sigmoid".
    If not specified, a linear activation function is used.
    '''
    if function == "relu":
        return relu(x)
    elif function == "sigmoid":
        return sigmoid(x)
    else:
        return x


def readCleanedDataset(path):
    '''
    Read the final preprocessed dataset from path
    '''
    df = pd.read_csv(path)
    TO_USE = [col for col in df.columns if col not in ["Result"]]
    X = df[TO_USE].values
    y = df['Result'].values
    return X, y


class NN:

    def __init__(self, layer_dimensions):
        '''
        Constructor for Neural Network class.
        It takes in the layer architecture as input as a list containing the neurons per layer.
        '''
        self.layer_dimensions = layer_dimensions
        self.parameters = dict()
        self.num_layers = len(layer_dimensions)
        self.num_features = None
        self.v = dict()
        self.s = dict()

    def initializeParameters(self):
        '''
        Initialise the weights and bias associated with each layer, along with the moments for Adam optimisation. 
        '''
        L = len(self.layer_dimensions)
        for i in range(1, L):
            self.parameters["W"+str(i)] = getLayerWeight(self.layer_dimensions[i],
                                                         self.layer_dimensions[i-1], mode="xavier")
            self.parameters["b" +
                            str(i)] = getLayerBias(self.layer_dimensions[i], mode="small")
            self.v["dW"+str(i)] = np.zeros_like(self.parameters["W"+str(i)])
            self.v["db"+str(i)] = np.zeros_like(self.parameters["b"+str(i)])
            self.s["dW"+str(i)] = np.zeros_like(self.parameters["W"+str(i)])
            self.s["db"+str(i)] = np.zeros_like(self.parameters["b"+str(i)])

    def forwardPropagationLayer(self, A_prev, W, b, activation):
        '''
        Perform forward propagation for a single layer
        '''
        Z = np.add(np.dot(W, A_prev), b)
        if activation == "sigmoid":
            A = 1/(1+np.exp(-Z))
        if activation == "relu":
            A = np.maximum(0, Z)
        cache = (A_prev, W, b, Z)
        return A, cache

    def forwardPropagation(self, X):
        '''
        Perform forward propagation for the entire network
        '''
        caches = list()
        L = len(self.layer_dimensions) - 1
        A = X

        for l in range(1, L):
            A_prev = A
            A, cache = self.forwardPropagationLayer(
                A_prev, self.parameters["W"+str(l)], self.parameters["b"+str(l)], activation="relu")
            caches.append(cache)

        y_hat, cache = self.forwardPropagationLayer(
            A, self.parameters["W"+str(l+1)], self.parameters["b"+str(l+1)], activation="sigmoid")
        caches.append(cache)

        return y_hat, caches

    def computeCost(self, y_hat, y, regularize=False, l=0.1):
        '''
        Computes the loss associated with a set of predictions. Regularization parameters can be set accordingly
        to utilise L2 regularization loss
        '''
        m = y.shape[1]
        cost = np.add(np.multiply(y, np.log(y_hat)),
                      np.multiply(1-y, np.log(1-y_hat)))
        cost = (-1/m) * np.sum(cost)

        if regularize:
            return cost + (l/(2*m)) * self.computeRegularizationCost()
        return cost

    def computeRegularizationCost(self):
        '''
        Compute the L2 Regularization parameter using the weights
        '''
        regularization_cost = 0
        L = len(self.layer_dimensions) - 1
        for i in range(1, L):
            regularization_cost += np.sum(
                np.square(self.parameters["W"+str(i)]))
        return regularization_cost

    def backwardPropagationLayer(self, dA, cache, activation):
        '''
        Perform backpropagation for a single layer of the network
        '''
        A_prev, W, b, Z = cache
        m = A_prev.shape[1]

        if activation == "relu":
            dZ = dA * gradRelu(Z)

        if activation == "sigmoid":
            dZ = dA * gradSigmoid(Z)

        dA_prev = np.dot(W.T, dZ)
        dW = (1/m) * np.dot(dZ, A_prev.T)
        db = (1/m) * np.sum(dZ)

        return dA_prev, dW, db

    def backwardPropagation(self, y_hat, Y, caches, regularize=False, l=0.1):
        '''
        Perform backpropagation for the entire network.
        Regularization parameters can be set accordingly to utilise L2 regularization.
        '''
        L = len(self.layer_dimensions) - 1
        grads = dict()

        dAL = -(np.divide(Y, y_hat) - np.divide(1-Y, 1-y_hat))
        grads["dA"+str(L)], grads["dW"+str(L)], grads["db"+str(L)
                                                      ] = self.backwardPropagationLayer(dAL, caches[-1], activation="sigmoid")

        if regularize:
            grads["dW"+str(L)] += l*caches[-1][1]/caches[-1][0].shape[1]

        for i in list(reversed(range(L-1))):
            current_cache = caches[i]
            dA_prev, dW, db = self.backwardPropagationLayer(
                grads["dA"+str(i+2)], current_cache, activation="relu")

            if regularize:
                dW += l*current_cache[1]/current_cache[0].shape[1]

            grads["dA"+str(i+1)] = dA_prev
            grads["dW"+str(i+1)] = dW
            grads["db"+str(i+1)] = db
        return grads

    def updateParameters(self, gradients):
        '''
        Perform Gradient Descent to update weights and bias for all the layers
        '''
        L = len(self.parameters)//2

        for l in range(L):
            self.parameters["W"+str(l+1)] = self.parameters["W"+str(l+1)] - \
                self.learning_rate * gradients["dW"+str(l+1)]
            self.parameters["b"+str(l+1)] = self.parameters["b"+str(l+1)] - \
                self.learning_rate * gradients["db"+str(l+1)]

    def AdamUpdate(self, t, grads, beta1=0.9, beta2=0.999,  epsilon=1e-8):
        '''
        Perform Gradient Descent with Adam Optimisation to update weights and bias for all the layers
        '''
        L = len(self.parameters) // 2
        v_corrected = {}
        s_corrected = {}
        for l in range(L):
            self.v["dW" + str(l+1)] = beta1*self.v["dW"+str(l+1)
                                                   ] + (1-beta1)*grads["dW"+str(l+1)]
            self.v["db" + str(l+1)] = beta1*self.v["db"+str(l+1)
                                                   ] + (1-beta1)*grads["db"+str(l+1)]
            v_corrected["dW" + str(l+1)] = self.v["dW"+str(l+1)]/(1-beta1**t)
            v_corrected["db" + str(l+1)] = self.v["db"+str(l+1)]/(1-beta1**t)

            self.s["dW" + str(l+1)] = beta2*self.s["dW"+str(l+1)] + \
                (1-beta2)*np.square(grads["dW"+str(l+1)])
            self.s["db" + str(l+1)] = beta2*self.s["db"+str(l+1)] + \
                (1-beta2)*np.square(grads["db"+str(l+1)])

            s_corrected["dW" + str(l+1)] = self.s["dW"+str(l+1)]/(1-beta2**t)
            s_corrected["db" + str(l+1)] = self.s["db"+str(l+1)]/(1-beta2**t)

            self.parameters["W" + str(l+1)] = self.parameters["W"+str(l+1)]-(self.learning_rate)*(
                v_corrected["dW"+str(l+1)]/(np.sqrt(s_corrected["dW"+str(l+1)])+epsilon))
            self.parameters["b" + str(l+1)] = self.parameters["b"+str(l+1)]-(self.learning_rate)*(
                v_corrected["db"+str(l+1)]/(np.sqrt(s_corrected["db"+str(l+1)])+epsilon))

    def fit(self, X, y, learning_rate=0.001, epochs=50, regularize=False, l=0.1, initialize=False):
        '''
        Fit a model with X and y data to initiate training
        '''
        costs = list()
        self.num_features = X.shape[1]
        self.num_layers += 1
        self.layer_dimensions.insert(0, self.num_features)
        X = X.T
        y = np.reshape(y, (1, -1))
        self.learning_rate = learning_rate
        if initialize:
            self.initializeParameters()
        for i in range(epochs):
            y_hat, caches = self.forwardPropagation(X)
            cost = self.computeCost(y_hat, y, l=l, regularize=regularize)
            grads = self.backwardPropagation(
                y_hat, y, caches, l=l, regularize=regularize)
            self.AdamUpdate(t=i+1, grads=grads)
            if not (i+1) % 50:
                print(f"Cost at iteration {i+1}: {cost}")
            costs.append(cost)

    def predict(self, X):
        '''
        Return the predictions for input data X
        '''
        m = X.shape[1]
        X = X.T

        y_hat, caches = self.forwardPropagation(X)
        y_hat = y_hat.reshape(-1)

        predicted = (y_hat > 0.6).astype(int)
        return predicted

    def CM(self, y_test, y_test_obs):
        '''
        Prints confusion matrix
        y_test is list of y values in the test dataset
        y_test_obs is list of y values predicted by the model
        '''
        y_test = y_test.flatten()
        y_test_obs = y_test_obs.flatten()

        for i in range(len(y_test_obs)):
            if(y_test_obs[i] > 0.6):
                y_test_obs[i] = 1
            else:
                y_test_obs[i] = 0

        cm = [[0, 0], [0, 0]]
        fp = 0
        fn = 0
        tp = 0
        tn = 0

        for i in range(len(y_test)):
            if(y_test[i] == 1 and y_test_obs[i] == 1):
                tp = tp+1
            if(y_test[i] == 0 and y_test_obs[i] == 0):
                tn = tn+1
            if(y_test[i] == 1 and y_test_obs[i] == 0):
                fp = fp+1
            if(y_test[i] == 0 and y_test_obs[i] == 1):
                fn = fn+1
        cm[0][0] = tn
        cm[0][1] = fp
        cm[1][0] = fn
        cm[1][1] = tp

        p = tp/(tp+fp)
        r = tp/(tp+fn)
        f1 = (2*p*r)/(p+r)

        print("Confusion Matrix : ")
        print(cm)
        print("\n")
        print(f"Accuracy : {(tp + tn)/(tp + tn + fp + fn)}")
        print(f"Precision : {p}")
        print(f"Recall : {r}")
        print(f"F1 SCORE : {f1}")


#   Read Preprocessed Dataset
X, y = readCleanedDataset("../data/[CLEANED]LBW_Dataset.csv")

#   Split dataset into train and test sets with a test size = 33%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=0, stratify=None)

# Create model with an architecture of 3 layers with 128, 32 and 1 neuron each
model = NN([128, 32, 1])

#   fit the model on training data with tuned parameters
#   learning_rate = 0.0001, epochs = 1000, lambda = 0.001
model.fit(
    X_train,
    y_train,
    learning_rate=0.0001,
    epochs=1000,
    initialize=True,
    regularize=True,
    l=0.001
)

# Obtain predictions for test dataset
predictions = model.predict(X_test)

# Displaying performance metrics
print("\nTest Accuracy:\n")
model.CM(y_test, predictions)
