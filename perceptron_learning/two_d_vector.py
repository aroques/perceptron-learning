"""
Functions that operate on 2d vectors.

w0 (or x0) is a bias "dummy" weight,
so even though the vector is 3 dimensional,
we call it a 2 dimensional vector.

"""

import numpy as np


def get_perpendicular_vector(w):
    # Two lines are perpendicular if: m1 * m2 = -1.
    # The two slopes must be negative reciprocals of each other.
    m1 = get_slope(w)
    m2 = -1 / m1

    # m2 = - w[1] / w[2]
    return np.array([w[0], -1 * m2, 1])


def get_line(w, x_bound):
    x_range = np.array(range(-x_bound, x_bound))

    # Formula for line is: w1x1 + w2x2 + w0 = 0
    # we let x2 = y, and x1 = x, then solve for y = mx + b
    slope = get_slope(w)
    y_intercept = get_y_intercept(w)
    y_line = (slope * x_range) + y_intercept

    return x_range, y_line


def get_y_intercept(w):
    return - w[0] / w[2]


def get_slope(w):
    return - w[1] / w[2]
