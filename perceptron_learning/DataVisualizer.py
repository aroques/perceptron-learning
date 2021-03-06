import numpy as np
import matplotlib.pyplot as plt
from . import two_d_vector as tdv


class DataVisualizer:
    def __init__(self, title, subtitle, x_bound, y_bound):
        plt.style.use('seaborn-whitegrid')
        self.fig, self.ax = plt.subplots()
        self.title = title
        self.subtitle = subtitle
        self.x_bound = x_bound
        self.y_bound = y_bound

    def setup_axes(self):
        self.ax.cla()
        self.fig.canvas.set_window_title(self.subtitle)
        self.fig.suptitle(self.title, fontsize=18)
        self.ax.set_title(self.subtitle, fontsize=14)
        self.ax.set_xlim(-self.x_bound, self.x_bound)
        self.ax.set_ylim(-self.y_bound, self.y_bound)

    @staticmethod
    def red_pts_above_line(pts, w_target, true_classes):
        pt_above_line = tdv.pt_above_line(pts[0, :], w_target)

        pt_is_positive_class = true_classes[0] > 0

        if pt_above_line and pt_is_positive_class:
            # positive pt above line
            return True
        if not pt_above_line and not pt_is_positive_class:
            # negative pt below line
            return True

        return False

    def plot_hypothesis(self, pts, true_classes, w_hypothesis, w_target=None):
        self.setup_axes()

        self.ax.scatter(x=pts[:, 0], y=pts[:, 1], marker='x',
                        color=['r' if sign >= 0 else 'b' for sign in true_classes])

        if w_target is not None:
            x, y = tdv.get_line(w_target, self.x_bound)
            self.ax.plot(x, y, label='target', color='m')

        x, y = tdv.get_line(w_hypothesis, self.x_bound)
        self.ax.plot(x, y, label='hypothesis', color='g')

        if w_target is not None:
            if self.red_pts_above_line(pts, w_target, true_classes):
                self.ax.fill_between(x, y, np.full((1,), self.y_bound), color=(1, 0, 0, 0.15))
                self.ax.fill_between(x, y, np.full((1,), -self.y_bound), color=(0, 0, 1, 0.15))
            else:
                self.ax.fill_between(x, y, np.full((1,), self.y_bound), color=(0, 0, 1, 0.15))
                self.ax.fill_between(x, y, np.full((1,), -self.y_bound), color=(1, 0, 0, 0.15))

        self.ax.legend(facecolor='w', fancybox=True, frameon=True, edgecolor='black', borderpad=1)

        # plt.pause(0.01)

    @staticmethod
    def visualize():
        plt.show()
