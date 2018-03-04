import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5,
                                       (self.input_nodes, self.hidden_nodes)) # weights_input_to_hidden.shape = (s_1,s_2)

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5,
                                       (self.hidden_nodes, self.output_nodes)) # weights_hidden_to_output.shape = (s_2,s_3)
        self.lr = learning_rate

        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        self.activation_function = lambda x : 1/(1+np.exp(-x))  # Replace 0 with your sigmoid calculation.

        ### If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your
        # implementation there instead.
        #
        #def sigmoid(x):
        #    return 0  # Replace 0 with your sigmoid calculation here
        #self.activation_function = sigmoid


    def train(self, features, targets):
        ''' Train the network on batch of features and targets.

            Arguments
            ---------

            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values

        '''
        n_records = features.shape[0] # features.shape = (M,n), n_records = M
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape) # delta_weights_i_h.shape = (s_1,s_2)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape) # delta_weights_h_o.shape = (s_2,s_3)
        for X, y in zip(features, targets): # X.shape = (n,), y.shape = (s_L,) = (s_3,)

            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y,
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        ''' Implement forward pass here

            Arguments
            ---------
            X: features batch

        '''
        #### Implement the forward pass here ####
        ### Forward pass ###

        # Useful shapes:
        # X.shape = (n,)
        # X[None,:].shape = (1,n) = (1,s_1)
        # self.weights_input_to_hidden.shape = (s_1,s_2)
        # self.weights_hidden_to_output.shape = (s_2,s_3)

        # TODO: Hidden layer - Replace these values with your calculations.

        # (1,s_1) x (s_1,s_2) = (1,s_2)
        hidden_inputs = np.matmul(X[None,:],self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        # hidden_inputs.shape = hidden_outputs.shape = (1,s_2)

        # TODO: Output layer - Replace these values with your calculations.

        # (1,s_2) x (s_2,s_3) = (1,s_3)
        final_inputs = np.matmul(hidden_outputs,self.weights_hidden_to_output) # signals into final output layer
        # final_outputs = self.activation_function(final_inputs)
        final_outputs = final_inputs # signals from final output layer, do not apply sigmoid as final layer has activation function f(x)=x
        # final_inputs.shape = final_outputs.shape = (1,s_3)

        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation

            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        #### Implement the backward pass here ####
        ### Backward pass ###

        # Useful shapes
        # hidden_outputs = (1,s_2)
        # final_outputs.shape = (1,s_3)
        # X.shape = (n,)
        # X[None,:].shape = (1,n) = (1,s_1)
        # y.shape = (s_L,) = (s_3,)
        # y[None, :].shape = (1,s_L) = (1,s_3)
        # self.weights_input_to_hidden.shape = (s_1,s_2)
        # self.weights_hidden_to_output.shape = (s_2,s_3)
        # delta_weights_i_h.shape = (s_1,s_2)
        # delta_weights_h_o.shape = (s_2,s_3)

        # TODO: Output error - Replace this value with your calculations.
        # (1,s_3) - (1,s_3) = (1,s_3)
        try: # for when y is a np.ndarray with shape = (1,), specifically in the unit tests
            error = y[None,:] - final_outputs # Output layer error is the difference between desired target and actual output.
        except: # for when y is simply a float, specifically when training network from pandas df
            error = np.array([[y]]) - final_outputs
        # error.shape = (1,s_3)

        # TODO: Calculate the hidden layer's contribution to the error
        # (1,s_3) x (s_2,s_3).T = (1,s_3) x (s_3,s_2) = (1,s_2)
        hidden_error = np.matmul(error,self.weights_hidden_to_output.T)
        # hidden_error.shape = (1,s_2)

        # TODO: Backpropagated error terms - Replace these values with your calculations.
        # (1,s_3) * (1,s_3) * (1 - (1,s_3)) = (1,s_3)
        # output_error_term = error * final_outputs * (1 - final_outputs)
        output_error_term = error # do not multiply by activations because the activation of final layer is simply f(x)=x
        # output_error_term.shape = (1,s_3)

        # (1,s_2) * (1,s_2) * (1 - (1,s_2)) = (1,s_2)
        hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)
        # hidden_error_term.shape = (1,s_2)

        # Weight step (input to hidden)
        # (1,s_1).T x (1,s_2) = (s_1,1) x (1,s_2) = (s_1,s_2)
        delta_weights_i_h += np.matmul(X[None,:].T, hidden_error_term)
        # delta_weights_i_h.shape = (s_1,s_2)

        # Weight step (hidden to output)
        # (1,s_2).T x (1,s_3) = (s_2,1) x (1,s_3) = (s_2,s_3)
        delta_weights_h_o += np.matmul(hidden_outputs.T, output_error_term)
        # delta_weights_h_o.shape = (s_2,s_3)

        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step

            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records # update input-to-hidden weights with gradient descent step

    def run(self, features):
        ''' Run a forward pass through the network with input features

            Arguments
            ---------
            features: 1D array of feature values
        '''

        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        hidden_inputs = np.matmul(features,self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

        # TODO: Output layer - Replace these values with the appropriate calculations.
        final_inputs = np.matmul(hidden_outputs,self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer, do not apply sigmoid as final layer has activation function f(x)=x

        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 1600
learning_rate = 1
hidden_nodes = 7
output_nodes = 1
