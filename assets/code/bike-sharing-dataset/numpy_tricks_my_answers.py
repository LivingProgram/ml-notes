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

        '''
        A numpy trick :
            if p.shape = (a,), q.shape = (a,b)
            np.dot(p,q).shape = (b,)

        Useful shapes :
            X.shape = (n,) = (s_1,)
            self.weights_input_to_hidden.shape = (s_1,s_2)
            self.weights_hidden_to_output.shape = (s_2,s_3)
        '''

        #### Implement the forward pass here ####
        ### Forward pass ###

        # TODO: Hidden layer - Replace these values with your calculations.
        # (s_1,) x (s_1,s_2) = (s_2,)
        hidden_inputs = np.dot(X,self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        # hidden_inputs.shape = hidden_outputs.shape = (s_2,)

        # TODO: Output layer - Replace these values with your calculations.
        # (s_2,) x (s_2,s_3) = (s_3,)
        final_inputs = np.dot(hidden_outputs,self.weights_hidden_to_output) # signals into final output layer
        # final_outputs = self.activation_function(final_inputs)
        final_outputs = final_inputs # signals from final output layer, do not apply sigmoid as final layer has activation function f(x)=x
        # final_inputs.shape = final_outputs.shape = (s_3,)

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

        '''
        Useful shapes :
            hidden_outputs = (s_2,)
            final_outputs.shape = (s_3,)
            X.shape = (n,) = (s_1,)
            y.shape = (s_L,) = (s_3,)
            self.weights_input_to_hidden.shape = (s_1,s_2)
            self.weights_hidden_to_output.shape = (s_2,s_3)
            delta_weights_i_h.shape = (s_1,s_2)
            delta_weights_h_o.shape = (s_2,s_3)
        '''

        #### Implement the backward pass here ####
        ### Backward pass ###

        # TODO: Output error - Replace this value with your calculations.
        # (s_3,) - (s_3,) = (s_3,)
        error = y - final_outputs # universally compatible, even if y is a float coming from a pandas df
        # error.shape = (s_3,)

        # TODO: Calculate the hidden layer's contribution to the error
        # (s_2,s_3) x (s_3,) = (s_2,)
        hidden_error = np.dot(self.weights_hidden_to_output,error)
        # hidden_error.shape = (s_2,)

        # TODO: Backpropagated error terms - Replace these values with your calculations.
        # (s_3,) * (s_3,) * (1 - (s_3,)) = (s_3,)
        # output_error_term = error * final_outputs * (1 - final_outputs)
        output_error_term = error # do not multiply by activations because the activation of final layer is simply f(x)=x
        # output_error_term.shape = (s_3,)

        # (s_2,) * (s_2,) * (1 - (s_2,)) = (s_2,)
        hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)
        # hidden_error_term.shape = (s_2,)

        '''
        A numpy trick :
            if p = (a,) and q = (b,)
            then p * q[:,None] = (a,) * (b,)[:,None] = (a,) * (b,1) = (b,a)

        The trick is essentially the same thing as :
            np.dot(q[:,None],p[:,None].T)
            = np.dot((b,)[:,None],(a,)[:,None].T)
            = np.dot((b,1),(a,1).T)
            = np.dot((b,1),(a,1).T) = (b,a)
        '''

        # Weight step (input to hidden)
        # (s_2,) * (s_1,)[:,None] = (s_2,) * (s_1,1) = (s_1,s_2)
        delta_weights_i_h += hidden_error_term * X[:,None]
        # delta_weights_i_h.shape = (s_1,s_2)

        # Weight step (hidden to output)
        # hidden_outputs = (s_2,), output_error_term.shape =
        # (s_3,) * (s_2,)[:,None] = (s_3,) * (s_2,1) = (s_2,s_3)
        delta_weights_h_o += output_error_term * hidden_outputs[:,None]
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

        '''
        A numpy trick :
            if p.shape = (a,), q.shape = (a,b)
            np.dot(p,q).shape = (b,)

        Useful shapes :
            features.shape = (n,) = (s_1,)
            self.weights_input_to_hidden.shape = (s_1,s_2)
            self.weights_hidden_to_output.shape = (s_2,s_3)
        '''

        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        # (s_1,) x (s_1,s_2) = (s_2,)
        hidden_inputs = np.dot(features,self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        # hidden_inputs.shape = hidden_outputs.shape = (s_2,)

        # TODO: Output layer - Replace these values with the appropriate calculations.
        # (s_2,) x (s_2,s_3) = (s_3,)
        final_inputs = np.dot(hidden_outputs,self.weights_hidden_to_output) # signals into final output layer
        # final_outputs = self.activation_function(final_inputs)
        final_outputs = final_inputs # signals from final output layer, do not apply sigmoid as final layer has activation function f(x)=x
        # final_inputs.shape = final_outputs.shape = (s_3,)

        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 1888
learning_rate = 1
hidden_nodes = 7
output_nodes = 1
