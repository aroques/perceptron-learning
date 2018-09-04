"""
Logs statistics
"""


class Logger:
    def __init__(self):
        self.num_iterations = 0
        self.num_vector_updates = 0

    def print_statistics(self):
        print('{:24s}: {:}'.format('Number of iterations', self.num_iterations))
        print('{:24s}: {:}'.format('Number of vector updates', self.num_vector_updates))
