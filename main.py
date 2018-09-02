import numpy as np
import matplotlib.pyplot as plt

x_bound = y_bound = bound = 100
num_points = 50


def main():
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots()

    pts = np.random.randint(-bound, bound, size=(num_points, 2))
    x_vectors = np.insert(pts, 0, 1, axis=1)  # Let x0 equal 1

    w_hypothesis = np.random.uniform(-10, 10, 3)
    w_target = get_perpendicular_vect(w_hypothesis)

    true_classes = np.sign(np.dot(x_vectors, w_target))

    pred_classes = np.sign(np.dot(x_vectors, w_hypothesis))
    misclassified_pts = np.not_equal(pred_classes, true_classes)

    alpha = 0.005  # step size

    while np.sum(misclassified_pts) > 0:
        for i, misclassified_pt in enumerate(np.nditer(misclassified_pts)):
            if misclassified_pt:
                # update rule: w(t + 1) = w(t) + y(t) * x(t) * alpha
                w_hypothesis += true_classes[i] * x_vectors[i] * alpha

                plot_hypothesis(ax, pts, true_classes, w_hypothesis, w_target)

        pred_classes = np.sign(np.dot(x_vectors, w_hypothesis))
        misclassified_pts = np.not_equal(pred_classes, true_classes)

    plt.show()


def setup_axes(ax):
    ax.cla()
    ax.set_title("Perceptron Learning")
    ax.set_xlim(-x_bound, x_bound)
    ax.set_ylim(-y_bound, y_bound)


def get_perpendicular_vect(w):
    # Two lines are perpendicular if: m1 * m2 = -1.
    # The two slopes must be negative reciprocals of each other.
    m1 = - w[1] / w[2]
    m2 = -1 / m1

    # m2 = - w[1] / w[2]
    return np.array([w[0], -1 * m2, 1])


def plot_hypothesis(ax, pts, true_classes, w_hypothesis, w_target):
    setup_axes(ax)

    ax.scatter(x=pts[:, 0], y=pts[:, 1], marker='x',
               color=['r' if sign >= 0 else 'b' for sign in true_classes])

    x, y = get_line(w_target)
    ax.plot(x, y, label='target', color='m')

    x, y = get_line(w_hypothesis)
    ax.plot(x, y, label='hypothesis', color='g')

    ax.fill_between(x, y, np.full((1,), y_bound), color=(1, 0, 0, 0.15))
    ax.fill_between(x, y, np.full((1,), -y_bound), color=(0, 0, 1, 0.15))

    ax.legend(facecolor='w', fancybox=True, frameon=True, edgecolor='black', borderpad=1)

    plt.pause(0.01)


def get_line(w):
    x_range = np.array(range(-x_bound, x_bound))

    # Formula for line is: w1x1 + w2x2 + w0 = 0
    # we let x2 = y, and x1 = x, then solve for y = mx + b
    slope = - w[1] / w[2]
    intercept = - w[0] / w[2]
    y_line = (slope * x_range) + intercept

    return x_range, y_line


if __name__ == '__main__':
    main()
