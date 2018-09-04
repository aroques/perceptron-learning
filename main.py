import numpy as np
from perceptron_learning \
    import Logger, two_d_vector as tdv, DataVisualizer

x_bound = y_bound = bound = 100
num_points = 50


def main():

    dv = DataVisualizer('Perceptron Learning', x_bound, y_bound)

    pts = np.random.randint(-bound, bound, size=(num_points, 2))
    x_vectors = np.insert(pts, 0, 1, axis=1)  # Let x0 equal 1

    w_hypothesis = np.random.uniform(-10, 10, 3)
    w_target = tdv.get_perpendicular_vector(w_hypothesis)

    alpha = 0.003

    true_classes = np.sign(np.dot(x_vectors, w_target))

    misclassified_pts = predict_and_evaluate(true_classes, w_hypothesis, x_vectors)

    logger = Logger()

    while np.sum(misclassified_pts) > 0:
        for i, misclassified_pt in enumerate(np.nditer(misclassified_pts)):
            if misclassified_pt:
                # update rule: w(t + 1) = w(t) + y(t) * x(t) * alpha
                w_hypothesis += true_classes[i] * x_vectors[i] * alpha
                logger.num_vector_updates += 1

                dv.plot_hypothesis(pts, true_classes, w_hypothesis, w_target)

        misclassified_pts = predict_and_evaluate(true_classes, w_hypothesis, x_vectors)

        logger.num_iterations += 1

    logger.print_statistics(w_target, w_hypothesis)

    dv.visualize()


def predict_and_evaluate(true_classes, w_hypothesis, x_vectors):
    pred_classes = np.sign(np.dot(x_vectors, w_hypothesis))
    misclassified_pts = np.not_equal(pred_classes, true_classes)
    return misclassified_pts


if __name__ == '__main__':
    main()
