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

        self.w_hypothesis = np.random.uniform(-10, 10, 3)
        self.error = float('inf')

        self.logger = Logger()
        self.dv = None

    def fit(self, x_train, y_train=None, w_target=None):
        """Fits the model to the training data (class labels) or target function.

        :param x_train: the training data
        :param y_train: will be passed in in the non-linearly separable case
        :param w_target: will be passed in in the linearly separable case
        :return: None
        """
        self.w_hypothesis = np.random.uniform(-10, 10, 3)
        self.error = float('inf')

        if w_target is not None:
            y_train = np.sign(np.dot(x_train, w_target))
            self.w_hypothesis = tdv.get_perpendicular_vector(w_target)

        pts = x_train[:, 1:]

        self.dv = DataVisualizer('Perceptron Learning', x_bound, y_bound)

        hypothesis = self.w_hypothesis

        misclassified_pts = predict_and_evaluate(hypothesis, x_train, y_train)

        self.logger = Logger()

        while self.logger.num_vector_updates < 250 and np.sum(misclassified_pts) > 0:
            for i, misclassified_pt in enumerate(np.nditer(misclassified_pts)):
                if misclassified_pt:
                    # update rule: w(t + 1) = w(t) + y(t) * x(t) * alpha
                    hypothesis += (y_train[i] * x_train[i] * self.alpha)

                    self.dv.plot_hypothesis(pts, y_train, hypothesis, w_target)

                    self.logger.num_vector_updates += 1

                    these_misclassified_pts = predict_and_evaluate(hypothesis, x_train, y_train)

                    this_error = calculate_error(np.sum(these_misclassified_pts), x_train.shape[0])

                    if this_error < self.error:
                        print('this error: {} is less than {}'.format(this_error, self.error))
                        self.w_hypothesis = hypothesis
                        self.error = this_error

            misclassified_pts = predict_and_evaluate(hypothesis, x_train, y_train)

            self.logger.num_iterations += 1

        self.dv.plot_hypothesis(pts, y_train, self.w_hypothesis, w_target)

        self.print_fit_statistics()

    def print_fit_statistics(self):
        self.logger.print_statistics()

        print('{:24s}: y = {:.2f}x + {:.2f}'.format('Hypothesis',
                                                    tdv.get_slope(self.w_hypothesis),
                                                    tdv.get_y_intercept(self.w_hypothesis)))

    def visualize_training(self):
        self.dv.visualize()

    def predict(self, x):
        return predict(x, self.w_hypothesis)
