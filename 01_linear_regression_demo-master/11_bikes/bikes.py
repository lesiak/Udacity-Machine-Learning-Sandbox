import numpy as np

from data_prep import train_features, train_targets, val_features, val_targets


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.hidden_nodes ** -0.5,
                                                        (self.hidden_nodes, self.input_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.output_nodes ** -0.5,
                                                         (self.output_nodes, self.hidden_nodes))
        self.lr = learning_rate

        #### Set this to your implemented sigmoid function ####
        # Activation function is the sigmoid function
        self.activation_function = lambda x: 1 / (1 + np.exp(-x))

    def train(self, inputs_list, targets_list):
        # Convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        #### Implement the forward pass here ####
        ### Forward pass ###
        # Hidden layer
        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)          # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)              # signals from hidden layer

        # Output layer
        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)  # signals into final output layer
        # output layer activation function is identity
        final_outputs = final_inputs                                          # signals from final output layer

        #### Implement the backward pass here ####
        ### Backward pass ###
        # Output layer error is the difference between desired target and actual output.
        output_errors = targets - final_outputs
        # activation is linear, hence gradient is same as error
        output_grad = output_errors

        hidden_errors = np.dot(output_grad, self.weights_hidden_to_output)
        # hidden layer gradients
        hidden_grad = hidden_errors.T * hidden_outputs * (1 - hidden_outputs)

        # Update the weights
        self.weights_hidden_to_output += self.lr * np.dot(output_errors, hidden_outputs.T)
        self.weights_input_to_hidden += self.lr * np.dot(hidden_grad, inputs.T)

    def run(self, inputs_list):
        # Run a forward pass through the network
        inputs = np.array(inputs_list, ndmin=2).T

        #### Implement the forward pass here ####
        # Hidden layer
        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)          # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)              # signals from hidden layer

        # Output layer
        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)  # signals into final output layer
        final_outputs = final_inputs                                          # signals from final output layer

        return final_outputs


def MSE(y, Y):
    return np.mean((y - Y) ** 2)


import sys

### Set the hyperparameters here ###
epochs = 100
learning_rate = 0.25
hidden_nodes = 5
output_nodes = 1

N_i = train_features.shape[1]
network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)

losses = {'train': [], 'validation': []}
for e in range(epochs):
    # Go through a random batch of 128 records from the training data set
    batch = np.random.choice(train_features.index, size=128)
    for record, target in zip(train_features.ix[batch].values,
                              train_targets.ix[batch]['cnt']):
        network.train(record, target)

    # Printing out the training progress
    train_loss = MSE(network.run(train_features), train_targets['cnt'].values)
    val_loss = MSE(network.run(val_features), val_targets['cnt'].values)
    sys.stdout.write("\rProgress: " + str(100 * e / float(epochs))[:4] \
                     + "% ... Training loss: " + str(train_loss)[:5] \
                     + " ... Validation loss: " + str(val_loss)[:5])

    losses['train'].append(train_loss)
    losses['validation'].append(val_loss)


# inputs = [0.5, -0.2, 0.1]
# targets = [0.4]
# test_w_i_h = np.array([[0.1, 0.4, -0.3],
#                       [-0.2, 0.5, 0.2]])
# test_w_h_o = np.array([[0.3, -0.1]])

# network = NeuralNetwork(3, 2, 1, 0.5)
# network.weights_input_to_hidden = test_w_i_h.copy()
# network.weights_hidden_to_output = test_w_h_o.copy()

# network.train(inputs, targets)
# print(network.weights_hidden_to_output)

# print(network.weights_input_to_hidden)
