import numpy as np
import random
from perceptron_learning import Perceptron
from perceptron_learning import two_d_vector as tdv

x_bound = y_bound = bound = 100
num_points = 50


def main():

    perceptron = Perceptron(alpha=0.005)

    pts = np.random.randint(-bound, bound, size=(num_points, 2))

    x_train = np.insert(pts, 0, 1, axis=1)  # Let x0 equal 1

    w_target = np.random.uniform(-10, 10, 3)

    y_train = get_y_train(pts, w_target)

    perceptron.fit(x_train, y_train=y_train)

    #perceptron.fit(x_train, w_target=w_target)


def get_y_train(pts, w_target):
    # Have y_train be somewhat linearly separable
    y_train = np.random.randint(-1, 1, num_points)

    for i, pt in enumerate(pts):
        pct_chance = .55
        pt_above_line = tdv.pt_above_line(pt, w_target)

        if pt_above_line and random.random() < pct_chance:
            y_train[i] = 1
        if not pt_above_line and random.random() < pct_chance:
            y_train[i] = -1

    return y_train


if __name__ == '__main__':
    main()
