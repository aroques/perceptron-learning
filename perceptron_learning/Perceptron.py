import numpy as np
from . import two_d_vector as tdv
from . import DataVisualizer, Logger

x_bound = y_bound = bound = 100


class Perceptron:
    def __init__(self, alpha):
        self.alpha = alpha
        self.w_hypothesis = np.random.uniform(-10, 10, 3)
        self.logger = Logger()
        self.dv = None

    def fit(self, x_train, y_train=None, w_target=None):
        """Fits the model to the training data (class labels) or target function.

        :param x_train: the training data
        :param y_train: will be passed in in the non-linearly separable case
        :param w_target: will be passed in in the linearly separable case
        :return: None
        """
        if w_target is not None:
            y_train = np.sign(np.dot(x_train, w_target))
            self.w_hypothesis = tdv.get_perpendicular_vector(w_target)

        pts = x_train[:, 1:]

        self.dv = DataVisualizer('Perceptron Learning', x_bound, y_bound)

        misclassified_pts = self.predict_and_evaluate(x_train, y_train)

        self.logger = Logger()

        while self.logger.num_vector_updates < 150 and np.sum(misclassified_pts) > 0:
            for i, misclassified_pt in enumerate(np.nditer(misclassified_pts)):
                if misclassified_pt:
                    # update rule: w(t + 1) = w(t) + y(t) * x(t) * alpha
                    self.w_hypothesis += y_train[i] * x_train[i] * self.alpha
                    self.logger.num_vector_updates += 1

                    self.dv.plot_hypothesis(pts, y_train, self.w_hypothesis, w_target)

            misclassified_pts = self.predict_and_evaluate(x_train, y_train)

            self.logger.num_iterations += 1

        self.print_fit_statistics()

    def predict_and_evaluate(self, x_train, y_train):
        pred_classes = self.predict(x_train)
        misclassified_pts = np.not_equal(pred_classes, y_train)
        return misclassified_pts

    def print_fit_statistics(self):
        self.logger.print_statistics()

        print('{:24s}: y = {:.2f}x + {:.2f}'.format('Hypothesis',
                                                    tdv.get_slope(self.w_hypothesis),
                                                    tdv.get_y_intercept(self.w_hypothesis)))

    def visualize_training(self):
        self.dv.visualize()

    def predict(self, x):
        return np.sign(np.dot(x, self.w_hypothesis))

