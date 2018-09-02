"""
Logs statistics
"""

from . import weight_vector as wv


class Logger:
    def __init__(self):
        self.num_iterations = 0
        self.num_vector_updates = 0

    def print_statistics(self, w_target, w_hypothesis):
        print('target function: y = {0:.2f}x + {1:.2f}'.format(wv.get_slope(w_target), wv.get_y_intercept(w_target)))
        print('hypothesis : y = {0:.2f}x + {1:.2f}'.format(wv.get_slope(w_hypothesis), wv.get_y_intercept(w_hypothesis)))
        print('num iterations: ', self.num_iterations)
        print('num weight vector updates: ', self.num_vector_updates)
        # need final mis-classification error
