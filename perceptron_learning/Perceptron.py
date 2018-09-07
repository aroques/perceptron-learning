import numpy as np
from . import two_d_vector as tdv
from . import DataVisualizer, Logger

x_bound = y_bound = bound = 100


def predict_and_evaluate(hypothesis, x_train, y_train):
    pred_classes = predict(hypothesis, x_train)
    misclassified_pts = np.not_equal(pred_classes, y_train)
    return misclassified_pts


def predict(x, hypothesis):
    return np.sign(np.dot(x, hypothesis.T))


def calculate_error(num_misclassified_pts, num_pts):
    return num_misclassified_pts / float(num_pts)


class Perceptron:
    """Uses 'pocket' algorithm to keep best hypothesis in it's 'pocket'"""

    def __init__(self, alpha):
        self.alpha = alpha

        self.best_hypothesis = np.random.uniform(-10, 10, 3)
        self.lowest_error = float('inf')

        self.logger = Logger()
        self.dv = None

    def fit(self, x_train, y_train=None, w_target=None):
        """Fits the model to the training data (class labels) or target function.

        :param x_train: the training data
        :param y_train: will be passed in in the non-linearly separable case
        :param w_target: will be passed in in the linearly separable case
        :return: None
        """
        self.best_hypothesis = np.random.uniform(-10, 10, 3)
        self.lowest_error = float('inf')
        self.logger = Logger()
        self.dv = DataVisualizer('Perceptron Learning', x_bound, y_bound)

        if w_target is not None:
            y_train = np.sign(np.dot(x_train, w_target))
            self.best_hypothesis = tdv.get_perpendicular_vector(w_target)

        pts = x_train[:, 1:]

        hypothesis = self.best_hypothesis

        misclassified_pts = predict_and_evaluate(hypothesis, x_train, y_train)

        while self.logger.num_vector_updates < 100000 and np.sum(misclassified_pts) > 0:
            for i, misclassified_pt in enumerate(np.nditer(misclassified_pts)):
                if misclassified_pt:
                    # update rule: w(t + 1) = w(t) + y(t) * x(t) * alpha
                    hypothesis += y_train[i] * x_train[i] * self.alpha

                    these_misclassified_pts = predict_and_evaluate(hypothesis, x_train, y_train)

                    this_error = calculate_error(np.sum(these_misclassified_pts), x_train.shape[0])

                    if this_error < self.lowest_error:
                        self.best_hypothesis = hypothesis
                        self.lowest_error = this_error

                    self.logger.num_vector_updates += 1

            misclassified_pts = predict_and_evaluate(hypothesis, x_train, y_train)

            self.logger.num_iterations += 1

        self.dv.plot_hypothesis(pts, y_train, self.best_hypothesis, w_target)

        self.print_fit_statistics()

    def print_fit_statistics(self):
        self.logger.print_statistics()

        print('{:28s}: y = {:.2f}x + {:.2f}'.format('Hypothesis',
                                                    tdv.get_slope(self.best_hypothesis),
                                                    tdv.get_y_intercept(self.best_hypothesis)))
        print('{0:28s}: {1:.2f}%'.format('In Sample (Training) Error', self.lowest_error))

    def visualize_training(self):
        self.dv.visualize()

    def predict(self, x):
        return predict(x, self.best_hypothesis)
