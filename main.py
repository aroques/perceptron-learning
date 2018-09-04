import numpy as np
from perceptron_learning import Perceptron

x_bound = y_bound = bound = 100
num_points = 50


def main():

    perceptron = Perceptron(alpha=0.005)

    pts = np.random.randint(-bound, bound, size=(num_points, 2))

    x_train = np.insert(pts, 0, 1, axis=1)  # Let x0 equal 1

    w_target = np.random.uniform(-10, 10, 3)

    perceptron.fit(x_train, w_target)


if __name__ == '__main__':
    main()
