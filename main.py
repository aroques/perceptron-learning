import numpy as np
import random
from perceptron_learning import Perceptron
from perceptron_learning import two_d_vector as tdv


def main():
    bound = 100  # the value that the x and y values are bounded by
    num_pts = 80
    num_train_pts = 50

    perceptron = Perceptron(alpha=0.005)

    w_target = np.random.uniform(-10, 10, 3)

    x = get_random_x(num_pts, bound)

    x_train, x_test = x[:num_train_pts, :], x[num_train_pts:, :]
    y_test = np.sign(np.dot(x_test, w_target))

    print('---------- Linearly Separable Data ----------')
    perceptron.fit(x_train, w_target=w_target)
    predictions = perceptron.predict(x_test)
    print('{:28s}: y = {:.2f}x + {:.2f}'.format('Target Function',
                                                tdv.get_slope(w_target),
                                                tdv.get_y_intercept(w_target)))
    print_error(predictions, y_test)

    print()

    y = get_y(x[:, 1:], w_target)
    y_train, y_test = y[:num_train_pts], y[num_train_pts:]

    print('-------- Non-Linearly Separable Data --------')
    perceptron.fit(x_train, y_train=y_train)
    predictions = perceptron.predict(x_test)
    print_error(predictions, y_test)

    perceptron.visualize_training()


def print_error(predictions, y_test):
    error = np.sum(np.not_equal(predictions, y_test)) / y_test.shape[0]
    print('{0:28s}: {1:.2f}%'.format('Out of Sample (Test) Error', error * 100))


def get_y(training_pts, w_target):
    # Have y be somewhat linearly separable
    y = np.random.choice([-1, 1], training_pts.shape[0])

    for i, pt in enumerate(training_pts):
        pct_chance = .75
        pt_above_line = tdv.pt_above_line(pt, w_target)

        if pt_above_line and random.random() < pct_chance:
            y[i] = 1
        if not pt_above_line and random.random() < pct_chance:
            y[i] = -1

    return y


def get_random_x(num_points, bound):
    pts = get_random_pts(num_points, bound)
    x = np.insert(pts, 0, 1, axis=1)  # Let x0 equal 1
    return x


def get_random_pts(num_points, bound):
    return np.random.randint(-bound, bound, size=(num_points, 2))


if __name__ == '__main__':
    main()
