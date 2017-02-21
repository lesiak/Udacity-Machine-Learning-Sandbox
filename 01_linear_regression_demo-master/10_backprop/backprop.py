import numpy as np
from data_prep import features, targets, features_test, targets_test

np.random.seed(42)

def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))


# Derivative of the sigmoid function
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Hyperparameters
n_hidden = 3  # number of hidden units
epochs = 500
learnrate = 0.05

n_records, n_features = features.shape
last_loss = None
# Initialize weights
weights_input_hidden = np.random.normal(scale=1 / n_features ** .5,
                                        size=(n_features, n_hidden))
weights_hidden_output = np.random.normal(scale=1 / n_features ** .5,
                                         size=n_hidden)

for e in range(epochs):
    del_w_input_hidden = np.zeros(weights_input_hidden.shape)
    del_w_hidden_output = np.zeros(weights_hidden_output.shape)
    for x, y in zip(features.values, targets):
        ## Forward pass ##
        # TODO: Calculate the output
        hidden_input = np.dot(x, weights_input_hidden)
        hidden_activations = sigmoid(hidden_input)
        output = sigmoid(np.dot(hidden_activations, weights_hidden_output))

        ## Backward pass ##
        # TODO: Calculate the error
        error = y - output

        # TODO: Calculate error gradient in output unit
        #output_error_gradient = error * sigmoid_prime(hidden_activations)
        output_error_gradient = error * output * (1 - output)

        # TODO: propagate errors to hidden layer
        #hidden_error_gradient = np.dot(output_error_gradient, weights_hidden_output) * sigmoid_prime(hidden_input)
        hidden_error_gradient = np.dot(output_error_gradient, weights_hidden_output) * \
                       hidden_activations * (1 - hidden_activations)

        # TODO: Update the change in weights
        del_w_hidden_output += output_error_gradient * hidden_activations
        del_w_input_hidden += hidden_error_gradient * x[:, None]

    # TODO: Update weights
    weights_input_hidden += learnrate * del_w_input_hidden / float(n_records)
    weights_hidden_output += learnrate * del_w_hidden_output / float(n_records)

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        hidden_activations = sigmoid(np.dot(x, weights_input_hidden))
        out = sigmoid(np.dot(hidden_activations,
                             weights_hidden_output))
        loss = np.mean((out - targets) ** 2)

        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss

# Calculate accuracy on test data
hidden = sigmoid(np.dot(features_test, weights_input_hidden))
out = sigmoid(np.dot(hidden, weights_hidden_output))
predictions = out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))
