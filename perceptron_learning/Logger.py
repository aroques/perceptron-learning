"""
Logs statistics
"""

from . import two_d_vector as tdv


class Logger:
    def __init__(self):
        self.num_iterations = 0
        self.num_vector_updates = 0

    def print_statistics(self, w_target, w_hypothesis):
        print('{:24s}: y = {:.2f}x + {:.2f}'.format('target function',
                                                    tdv.get_slope(w_target), tdv.get_y_intercept(w_target)))
        print('{:24s}: y = {:.2f}x + {:.2f}'.format('hypothesis',
                                                    tdv.get_slope(w_hypothesis), tdv.get_y_intercept(w_hypothesis)))
        print('{:24s}: {:}'.format('number of iterations', self.num_iterations))
        print('{:24s}: {:}'.format('number of vector updates', self.num_vector_updates))
        # need final mis-classification error
