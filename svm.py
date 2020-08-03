import numpy as np

# Mathieu Blondel, October 2010
# License: BSD 3 clause
#
# orig src: https://gist.github.com/mblondel/656147
# modified for my use, Robert Collins, 2020


class KernelPerceptron(object):
    @staticmethod
    def linear_kernel(x1, x2):
        return x1 @ x2

    @staticmethod
    def polynomial_kernel(x, y, p=3):
        return (1 + x @ y) ** p

    @staticmethod
    def gaussian_kernel(x, y, sigma=5.0):
        return np.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

    def __init__(self, epochs=100, gamma=[(-1, -1), (1, 1)], kernel=None):
        self.kernel = kernel or KernelPerceptron.gaussian_kernel
        self.T = epochs
        self.gamma = dict(gamma)
        self.neg_gamma = dict([tuple(reversed(x)) for x in gamma])

    def train(self, X, y):
        y = np.array([self.neg_gamma[y_i] for y_i in y])
        n_samples, n_features = X.shape

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel(X[i], X[j])

        self.alpha = np.zeros(n_samples, dtype=np.float64)
        for t in range(self.T):
            for i in range(n_samples):
                ya = self.alpha * y
                if np.sign(np.sum(K[:, i] * ya)) != y[i]:
                    self.alpha[i] += 1.0

        # Support vectors
        sv = self.alpha > 1e-5
        self.alpha = self.alpha[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]

    def project(self, X):
        y_predict = np.zeros(len(X))
        for i in range(len(X)):
            s = 0
            for a, sv_y, sv in zip(self.alpha, self.sv_y, self.sv):
                s += a * sv_y * self.kernel(X[i], sv)
            y_predict[i] = s
        return y_predict

    def predict(self, X):
        X = np.atleast_2d(X)
        n_samples, n_features = X.shape
        return np.array([self.gamma[y] for y in np.sign(self.project(X))])
