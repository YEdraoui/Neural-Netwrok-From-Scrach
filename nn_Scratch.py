import numpy as np
from sklearn.model_selection import train_test_split

# Sigmoid Activation Function (used in the hidden and output layers)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Derivative of Sigmoid Function (used during backpropagation)
def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Initialize weights and biases for the layers of the network
def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * 0.01  # Small random values for weights
    b1 = np.zeros((n_h, 1))  # Zero initialization for biases
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

# Forward propagation: Calculate the activations and predictions
def forward_propagation(X, parameters):
    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]

    Z1 = np.dot(W1, X) + b1  # Linear combination for hidden layer
    A1 = sigmoid(Z1)  # Activation function applied to hidden layer
    Z2 = np.dot(W2, A1) + b2  # Linear combination for output layer
    A2 = sigmoid(Z2)  # Output prediction using sigmoid activation

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}  # Save intermediate values
    return A2, cache

# Calculate the cross-entropy loss
def compute_loss(A2, Y):
    m = Y.shape[1]  # Number of examples
    loss = -np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2)) / m  # Cross-entropy loss
    return loss

# Backpropagation: Calculate gradients to update weights and biases
def backward_propagation(X, Y, cache, parameters):
    m = X.shape[1]  # Number of training examples
    W2 = parameters["W2"]

    dZ2 = cache["A2"] - Y  # Error at output layer
    dW2 = np.dot(dZ2, cache["A1"].T) / m  # Gradient for W2
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m  # Gradient for b2

    dZ1 = np.dot(W2.T, dZ2) * sigmoid_derivative(cache["Z1"])  # Error at hidden layer
    dW1 = np.dot(dZ1, X.T) / m  # Gradient for W1
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m  # Gradient for b1

    gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return gradients

# Update parameters using gradient descent
def update_parameters(parameters, gradients, learning_rate):
    parameters["W1"] -= learning_rate * gradients["dW1"]
    parameters["b1"] -= learning_rate * gradients["db1"]
    parameters["W2"] -= learning_rate * gradients["dW2"]
    parameters["b2"] -= learning_rate * gradients["db2"]
    return parameters

# Training loop: Perform forward and backward passes, then update parameters
def train(X, Y, n_x, n_h, n_y, num_iterations, learning_rate):
    parameters = initialize_parameters(n_x, n_h, n_y)  # Initialize weights and biases

    for i in range(num_iterations):
        A2, cache = forward_propagation(X, parameters)  # Forward pass
        cost = compute_loss(A2, Y)  # Compute loss
        gradients = backward_propagation(X, Y, cache, parameters)  # Backward pass
        parameters = update_parameters(parameters, gradients, learning_rate)  # Update weights

        if i % 100 == 0:  # Print cost every 100 iterations to track progress
            print(f"Iteration {i}, Cost: {cost:.4f}")

    return parameters

# Predict outputs based on trained model
def predict(X, parameters):
    A2, _ = forward_propagation(X, parameters)  # Get predictions
    return (A2 > 0.5).astype(int)  # Convert probabilities to binary predictions

# Generate synthetic data for training and testing
np.random.seed(1)
X = np.random.randn(2, 500)  # Two features, 500 examples
y = (np.sum(X, axis=0) > 0).astype(int).reshape(1, 500)  # Binary labels based on sum of features

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X.T, y.T, test_size=0.2, random_state=42)
X_train, X_test = X_train.T, X_test.T  # Transpose to match (features, examples) format
y_train, y_test = y_train.T, y_test.T

# Train the neural network
parameters = train(X_train, y_train, n_x=2, n_h=4, n_y=1, num_iterations=1000, learning_rate=0.01)

# Test the model
predictions = predict(X_test, parameters)
accuracy = np.mean(predictions == y_test) * 100  # Calculate accuracy
print(f"Test Accuracy: {accuracy:.2f}%")
