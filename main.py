from kernel import LaplaceKernel
import numpy as np

SEED = 1717
np.random.seed(SEED)

def main():
    # Data X of shape (number of samples x number of features)
    # Labels y of shape (number of samples x number of classes)

    # Here is a toy dataset of the form y = w x
    n = 1000  # number of samples
    d = 50  # number of features
    c = 10  # number of output dimensions
    X = np.random.normal(size=(n, d))
    w = np.random.normal(size=(d, c))
    y = X @ w

    n_test = 573
    X_test = np.random.normal(size=(n_test, d))
    y_test = X_test @ w

    model = LaplaceKernel(L=10)
    model.fit(X, y, reg=0)
    predictions = model.predict(X_test)
    print("Test MSE: ", np.mean(np.square(predictions - y_test)))

if __name__ == "__main__":
    main()
