I. IMPLEMENTATION

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



II. LIST OF HYPERPARAMETERS

1.  The model initialiser takes in the number of layers as a list containing the neurons per layer.
For example, if we use 2 layers with 16 and 2 neurons, we create the model with: 
model = NN([16, 2])

2.  The fit() function takes the following hyperparameters
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



III. KEY FEATURES OF DESIGN

1.  The model works for an arbitrary number of layers and is capable of performing forward propagation as well as
    backpropagation for any given combination of layers and neurons per layer. This is owing to the fact that the 
    implementation uses the recursive backpropagation equation to store the partial derivatives of each successive layer
    and use these derivatives in the earlier layers: 
        dJ/dW[i] = dJ/dZ[i] * A[i-1]'
        dJ/db[i] =  SUM {dJ/dZ[i]}

2.  The model also utilises the Xavier method of initilisation of weights so that the values are sampled from a normal
    distribution and uses the Adam optimiser to speed up training. 

3. The model finally uses L2 regularization to prevent overfitting, since the dataset is imbalanced.



IV. IMPLEMENTATION BEYOND THE BASICS

1.  The layers use Xavier's initialisation metric (as presented in the paper Understanding the difficulty 
    of training deep feedforward neural networks. In Aistats, volume 9, pages 249–256, 2010) for the initial values
    of the weights to ensure that the variance of the activations are the same across every layer.

2.  We have also utitlised L2 regularization to prevent overfitting of the model and to scale down the weights using weight decay.

3.  We use Adam Optimisation (as presented in the paper Adam: A method for Stochastic Optimization, ICLR 2015)
    as the optimization algorithm to approach convergence faster and hence speed up training.

3.  The architecture defined also uses the recursive backpropagation approach/equation, and hence can adapt to any change 
    in the number of layers as well as the neurons per layer - as long as the first L-1 layers use a ReLu activation 
    function and the Lth layer uses a Sigmoid activation function.



V. INSTRUCTIONS TO EXECUTE FILES

    1.  Preprocess the dataset using: python3 preprocess.py
    2.  Run the training using: python3 neuralnetwork.py