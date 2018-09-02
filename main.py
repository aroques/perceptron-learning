import numpy as np
import matplotlib.pyplot as plt
from perceptron_learning import Logger
from perceptron_learning import weight_vector as wv

x_bound = y_bound = bound = 100
num_points = 50


def main():
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots()

    pts = np.random.randint(-bound, bound, size=(num_points, 2))
    x_vectors = np.insert(pts, 0, 1, axis=1)  # Let x0 equal 1

    w_hypothesis = np.random.uniform(-10, 10, 3)
    w_target = get_perpendicular_vector(w_hypothesis)

    true_classes = np.sign(np.dot(x_vectors, w_target))

    misclassified_pts = predict_and_evaluate(true_classes, w_hypothesis, x_vectors)

    alpha = 0.003  # step size

    logger = Logger()

    while np.sum(misclassified_pts) > 0:
        for i, misclassified_pt in enumerate(np.nditer(misclassified_pts)):
            if misclassified_pt:
                # update rule: w(t + 1) = w(t) + y(t) * x(t) * alpha
                w_hypothesis += true_classes[i] * x_vectors[i] * alpha
                logger.num_vector_updates += 1

                plot_hypothesis(ax, pts, true_classes, w_hypothesis, w_target)

        misclassified_pts = predict_and_evaluate(true_classes, w_hypothesis, x_vectors)

        logger.num_vector_updates += 1

    logger.print_statistics(w_target, w_hypothesis)

    plt.show()


def predict_and_evaluate(true_classes, w_hypothesis, x_vectors):
    pred_classes = np.sign(np.dot(x_vectors, w_hypothesis))
    misclassified_pts = np.not_equal(pred_classes, true_classes)
    return misclassified_pts


def setup_axes(ax):
    ax.cla()
    ax.set_title("Perceptron Learning")
    ax.set_xlim(-x_bound, x_bound)
    ax.set_ylim(-y_bound, y_bound)


def get_perpendicular_vector(w):
    # Two lines are perpendicular if: m1 * m2 = -1.
    # The two slopes must be negative reciprocals of each other.
    m1 = wv.get_slope(w)
    m2 = -1 / m1

    # m2 = - w[1] / w[2]
    return np.array([w[0], -1 * m2, 1])


def plot_hypothesis(ax, pts, true_classes, w_hypothesis, w_target):
    setup_axes(ax)

    ax.scatter(x=pts[:, 0], y=pts[:, 1], marker='x',
               color=['r' if sign >= 0 else 'b' for sign in true_classes])

    x, y = wv.get_line(w_target, x_bound)
    ax.plot(x, y, label='target', color='m')

    x, y = wv.get_line(w_hypothesis, x_bound)
    ax.plot(x, y, label='hypothesis', color='g')

    ax.fill_between(x, y, np.full((1,), y_bound), color=(1, 0, 0, 0.15))
    ax.fill_between(x, y, np.full((1,), -y_bound), color=(0, 0, 1, 0.15))

    ax.legend(facecolor='w', fancybox=True, frameon=True, edgecolor='black', borderpad=1)

    plt.pause(0.01)


if __name__ == '__main__':
    main()
