import numpy as np
import matplotlib.pyplot as plt
from . import two_d_vector as tdv


class DataVisualizer():
    def __init__(self, title, x_bound, y_bound):
        plt.style.use('seaborn-whitegrid')
        self.fig, self.ax = plt.subplots()
        self.title = title
        self.x_bound = x_bound
        self.y_bound = y_bound

    def setup_axes(self):
        self.ax.cla()
        self.ax.set_title(self.title)
        self.ax.set_xlim(-self.x_bound, self.x_bound)
        self.ax.set_ylim(-self.y_bound, self.y_bound)

    def plot_hypothesis(self, pts, true_classes, w_hypothesis, w_target):
        self.setup_axes()

        self.ax.scatter(x=pts[:, 0], y=pts[:, 1], marker='x',
                        color=['r' if sign >= 0 else 'b' for sign in true_classes])

        x, y = tdv.get_line(w_target, self.x_bound)
        self.ax.plot(x, y, label='target', color='m')

        x, y = tdv.get_line(w_hypothesis, self.x_bound)
        self.ax.plot(x, y, label='hypothesis', color='g')

        self.ax.fill_between(x, y, np.full((1,), self.y_bound), color=(1, 0, 0, 0.15))
        self.ax.fill_between(x, y, np.full((1,), -self.y_bound), color=(0, 0, 1, 0.15))

        self.ax.legend(facecolor='w', fancybox=True, frameon=True, edgecolor='black', borderpad=1)

        plt.pause(0.01)

    @staticmethod
    def visualize():
        plt.show()
