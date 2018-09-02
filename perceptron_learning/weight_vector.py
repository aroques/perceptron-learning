"""
Helper file that contain functions to help visualize weight vector
"""

import numpy as np


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
