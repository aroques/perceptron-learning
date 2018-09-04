import numpy as np
import random
from perceptron_learning import Perceptron
from perceptron_learning import two_d_vector as tdv

x_bound = y_bound = bound = 100
num_train_pts = 50
num_test_pts = 30


def main():
    perceptron = Perceptron(alpha=0.005)

    w_target = np.random.uniform(-10, 10, 3)

    x_train = get_random_x(num_train_pts)
    x_test = get_random_x(num_test_pts)
    y_test = np.sign(np.dot(x_test, w_target))

    print('---------- Linearly Separable Data ----------')
    perceptron.fit(x_train, w_target=w_target)
    predictions = perceptron.predict(x_test)
    print('{:24s}: y = {:.2f}x + {:.2f}'.format('Target Function',
                                                tdv.get_slope(w_target),
                                                tdv.get_y_intercept(w_target)))
    error = np.sum(np.not_equal(predictions, y_test)) / num_test_pts
    print('{0:24s}: {1:.2f}%'.format('Misclassification Error', error * 100))

    print()

    y_train = get_y_train(x_train[:, 1:], w_target)

    print('-------- Non-Linearly Separable Data --------')
    perceptron.fit(x_train, y_train=y_train)
    predictions = perceptron.predict(x_test)
    y_test = np.random.choice(y_train, num_test_pts)
    error = np.sum(np.not_equal(predictions, y_test)) / num_test_pts
    print('{0:24s}: {1:.2f}%'.format('Misclassification Error', error * 100))

    perceptron.visualize_training()


def get_y_train(pts, w_target):
    # Have y_train be somewhat linearly separable
    y_train = np.random.choice([-1, 1], num_train_pts)

    for i, pt in enumerate(pts):
        pct_chance = .80
        pt_above_line = tdv.pt_above_line(pt, w_target)

        if pt_above_line and random.random() < pct_chance:
            y_train[i] = 1
        if not pt_above_line and random.random() < pct_chance:
            y_train[i] = -1

    return y_train


def get_random_x(num_points):
    pts = np.random.randint(-bound, bound, size=(num_points, 2))
    x = np.insert(pts, 0, 1, axis=1)  # Let x0 equal 1
    return x


if __name__ == '__main__':
    main()
