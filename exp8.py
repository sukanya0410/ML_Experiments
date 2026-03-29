# -----------------------------
# Experiment No. 8
# Single Layer Perceptron (SLP)
# -----------------------------

import numpy as np

# -----------------------------
# Perceptron Class
# -----------------------------
class Perceptron:

    def __init__(self, learning_rate=0.01, n_epochs=100):
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None

    # Activation Function (Step Function)
    def activation(self, z):
        return np.where(z >= 0, 1, 0)

    # Training Function
    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Training loop
        for epoch in range(self.n_epochs):
            for i in range(n_samples):
                z = np.dot(X[i], self.weights) + self.bias
                y_pred = self.activation(z)

                error = y[i] - y_pred

                # Update weights and bias
                if error != 0:
                    self.weights += self.lr * error * X[i]
                    self.bias += self.lr * error

        print("Training Completed")
        print("Final Weights:", self.weights)
        print("Final Bias:", self.bias)

    # Prediction Function
    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.activation(z)


# -----------------------------
# Main Program (AND Gate Example)
# -----------------------------
if __name__ == "__main__":

    # Training Data (AND Logic)
    X_train = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    y_train = np.array([0, 0, 0, 1])

    # Create Perceptron Model
    model = Perceptron(learning_rate=0.1, n_epochs=10)

    # Train Model
    model.fit(X_train, y_train)

    # Testing
    print("\nTesting Perceptron (AND Gate):")
    for x in X_train:
        prediction = model.predict(x)
        print(f"Input: {x} -> Output: {prediction}")