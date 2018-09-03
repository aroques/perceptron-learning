import numpy as np
import matplotlib.pyplot as plt
from . import two_d_vector as tdv
from . import Logger

x_bound = y_bound = bound = 100
num_points = 50

def setup_axes(ax):
    ax.cla()
    ax.set_title('Perceptron Learning')
    ax.set_xlim(-x_bound, x_bound)
    ax.set_ylim(-y_bound, y_bound)


def plot_hypothesis(ax, pts, true_classes, w_hypothesis, w_target):
    setup_axes(ax)

    ax.scatter(x=pts[:, 0], y=pts[:, 1], marker='x',
               color=['r' if sign >= 0 else 'b' for sign in true_classes])

    x, y = tdv.get_line(w_target, x_bound)
    ax.plot(x, y, label='target', color='m')

    x, y = tdv.get_line(w_hypothesis, x_bound)
    ax.plot(x, y, label='hypothesis', color='g')

    ax.fill_between(x, y, np.full((1,), y_bound), color=(1, 0, 0, 0.15))
    ax.fill_between(x, y, np.full((1,), -y_bound), color=(0, 0, 1, 0.15))

    ax.legend(facecolor='w', fancybox=True, frameon=True, edgecolor='black', borderpad=1)

    plt.pause(0.01)


class Perceptron():
    def __init__(self, alpha):
        self.alpha = alpha
        self.w_hypothesis = np.random.uniform(-10, 10, 3)
        self.w_target = tdv.get_perpendicular_vector(self.w_hypothesis)

    def predict_target_fn(self, x_vectors):
        fig, ax = plt.subplots()

        true_classes = np.sign(np.dot(x_vectors, self.w_target))

        misclassified_pts = self.predict_and_evaluate(true_classes, self.w_hypothesis, x_vectors)

        logger = Logger()

        while np.sum(misclassified_pts) > 0:
            for i, misclassified_pt in enumerate(np.nditer(misclassified_pts)):
                if misclassified_pt:
                    # update rule: w(t + 1) = w(t) + y(t) * x(t) * alpha
                    self.w_hypothesis += true_classes[i] * x_vectors[i] * alpha
                    logger.num_vector_updates += 1

                    plot_hypothesis(ax, pts, true_classes, self.w_hypothesis, self.w_target)

            misclassified_pts = self.predict_and_evaluate(true_classes, self.w_hypothesis, x_vectors)

            logger.num_iterations += 1

        logger.print_statistics(self.w_target, self.w_hypothesis)

    @staticmethod
    def predict_and_evaluate(true_classes, w_hypothesis, x_vectors):
        pred_classes = np.sign(np.dot(x_vectors, w_hypothesis))
        misclassified_pts = np.not_equal(pred_classes, true_classes)
        return misclassified_pts
