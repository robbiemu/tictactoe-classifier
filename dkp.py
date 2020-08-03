import numpy as np


class DirectKernelPerceptron:
    @staticmethod
    def linear_error(P):
        return np.ones(len(P), dtype=np.float64)

    @staticmethod
    def bounded_linear_error(P):
        def ƒ(t):
            return -t if t < 0 else 0

        ones = np.ones(len(P), dtype=np.float64)
        return np.array([ƒ(t) for t in np.matmul(ones, np.linalg.inv(P))])

    @staticmethod
    def quadratic_error_Q1(P):
        def ƒ(t):
            return (t - 1)**2/2

        ones = np.ones(len(P), dtype=np.float64)
        return np.array([ƒ(t) for t in np.matmul(ones, np.linalg.inv(P))])

    @staticmethod
    def quadratic_error(P):
        def ƒ(t):
            return (t - 1)**2/2

        ones = np.ones(len(P), dtype=np.float64)
        I = np.identity(len(P))
        return np.array([ƒ(t) for t in np.matmul(ones, np.linalg.inv(I + P))])

    @staticmethod
    def bounded_quadratic_error(t):
        def ƒ(t):
            return (t - 1)**2/2

        ones = np.ones(len(P), dtype=np.float64)
        I = np.identity(len(P))
        return np.array([ƒ(t) for t in np.matmul(ones, np.linalg.inv(I + P))])

    @staticmethod
    def gaussian_kernel(x, y, sigma=np.inf):
        return np.exp(-np.linalg.norm(x-y)**2 / (sigma**2))

    def factory_kernel(self, sigma=5.0):
        def kernel(x, y):
            return DirectKernelPerceptron.gaussian_kernel(x, y, sigma)
        return kernel

    def __init__(self,
                 sigma=5.0,
                 error=DirectKernelPerceptron.bounded_quadratic_error):
        self.error = error
        self.sigma = sigma

    def train(self, training_inputs, labels):
        N = len(training_inputs)
        X = training_inputs
        d = labels
        K = self.factory_kernel(self.sigma)

        self.P = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                self.P[i, j] = d[i] * d[j] * (2 * K(X[i], X[j]) - 1)

            self.α = self.error(P)

    def predict(self, inputs):
        K = self.factory_kernel(self.sigma)
        return self.sgn(np.sum(self.α * len(inputs.columns) * K(inputs)))
