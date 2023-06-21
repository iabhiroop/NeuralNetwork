import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("Bank_Personal_Loan_Modelling.csv")
df = df.drop(["ID", "ZIP Code"], axis=1)


df = pd.get_dummies(df, columns=["Family", "Education", "Securities Account", "CD Account", "Online", "CreditCard"])


X = df.drop(["Personal Loan"], axis=1).values
Y = df["Personal Loan"].values.reshape(-1, 1)

X = (X - X.mean()) / X.std()


class NeuralNetwork:

    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.01, num_iterations=10000):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.params = {}
        self.grads = {}
        self.losses = []
        
    def initialize_parameters(self):
        np.random.seed(0)
        self.params['W1'] = np.random.randn(self.input_dim, self.hidden_dim) / np.sqrt(self.input_dim)
        self.params['b1'] = np.zeros((1, self.hidden_dim))
        self.params['W2'] = np.random.randn(self.hidden_dim, self.output_dim) / np.sqrt(self.hidden_dim)
        self.params['b2'] = np.zeros((1, self.output_dim))
        
    def relu(self, Z):
        return np.maximum(0, Z)
    
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    def forward_propagation(self, X):
        Z1 = np.dot(X, self.params['W1']) + self.params['b1']
        A1 = self.relu(Z1)
        Z2 = np.dot(A1, self.params['W2']) + self.params['b2']
        A2 = self.sigmoid(Z2)
        cache = (Z1, A1, Z2, A2)
        return A2, cache

    def compute_loss(self, Y, Y_pred):
        m = Y.shape[0]
        loss = (-1/m) * np.sum(Y * np.log(Y_pred) + (1-Y) * np.log(1-Y_pred))
        return loss

    def backward_propagation(self, X, Y, cache):
        m = X.shape[0]
        Z1, A1, Z2, A2 = cache
        dZ2 = A2 - Y
        dW2 = (1/m) * np.dot(A1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        dA1 = np.dot(dZ2, self.params['W2'].T)
        dZ1 = dA1 * (Z1 > 0)
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        self.grads['dW1'] = dW1
        self.grads['db1'] = db1
        self.grads['dW2'] = dW2
        self.grads['db2'] = db2
        
    def update_parameters(self):
        self.params['W1'] -= self.learning_rate * self.grads['dW1']
        self.params['b1'] -= self.learning_rate * self.grads['db1']
        self.params['W2'] -= self.learning_rate * self.grads['dW2']
        self.params['b2'] -= self.learning_rate * self.grads['db2']
        
    def fit(self, X, Y):
        self.initialize_parameters()
        for i in range(self.num_iterations):
            Y_pred, cache = self.forward_propagation(X)
            loss = self.compute_loss(Y, Y_pred)
            self.losses.append(loss)
            self.backward_propagation(X, Y, cache)
            self.update_parameters()
            if i % 1000 == 0:
                print(f"Iteration {i}: Loss={loss}")
                
    def predict(self, X):
        Y_pred, _ = self.forward_propagation(X)
        return np.round(Y_pred)

    def evaluate(self, X_test, Y_test):
        Y_pred = self.predict(X_test)
        acc = accuracy_score(Y_test, Y_pred)
        print(f"Accuracy: {acc}")


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
nn = NeuralNetwork(input_dim=X.shape[1], hidden_dim=10, output_dim=1, learning_rate=0.01, num_iterations=100000)
nn.fit(X_train, Y_train)

nn.evaluate(X_test, Y_test)
