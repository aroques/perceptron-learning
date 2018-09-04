import numpy as np
from . import two_d_vector as tdv
from . import DataVisualizer, Logger

x_bound = y_bound = bound = 100


def predict_and_evaluate(true_classes, w_hypothesis, x_vectors):
    pred_classes = np.sign(np.dot(x_vectors, w_hypothesis))
    misclassified_pts = np.not_equal(pred_classes, true_classes)
    return misclassified_pts


class Perceptron:
    def __init__(self, alpha):
        self.alpha = alpha
        self.w_hypothesis = None

    def fit(self, x_train, w_target):
        pts = x_train[:, 1:]

        dv = DataVisualizer('Perceptron Learning', x_bound, y_bound)

        self.w_hypothesis = tdv.get_perpendicular_vector(w_target)

        true_classes = np.sign(np.dot(x_train, w_target))

        misclassified_pts = predict_and_evaluate(true_classes, self.w_hypothesis, x_train)

        logger = Logger()

        while np.sum(misclassified_pts) > 0:
            for i, misclassified_pt in enumerate(np.nditer(misclassified_pts)):
                if misclassified_pt:
                    # update rule: w(t + 1) = w(t) + y(t) * x(t) * alpha
                    self.w_hypothesis += true_classes[i] * x_train[i] * self.alpha
                    logger.num_vector_updates += 1

                    dv.plot_hypothesis(pts, true_classes, self.w_hypothesis, w_target)

            misclassified_pts = predict_and_evaluate(true_classes, self.w_hypothesis, x_train)

            logger.num_iterations += 1

        logger.print_statistics(w_target, self.w_hypothesis)

        dv.visualize()
