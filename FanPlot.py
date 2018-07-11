import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.dates import num2date

class FanPlot:

    def __init__(self, nrange=75, nbeam=16, r0=180, dr=45, dtheta=3.24):
        # Set member variables
        self.nrange = nrange
        self.nbeam = nbeam
        self.r0 = r0
        self.dr = dr
        self.dtheta = dtheta
        # Initial angle (from X, polar coordinates) for beam 0
        self.theta0 = (90 - dtheta * nbeam / 2)
        self._open_figure()


    def _open_figure(self):
        # Create axis
        self.fig = plt.figure(figsize=(12,9))
        self.ax = self.fig.add_subplot(111, polar=True)

        # Set up ticks and labels
        self.r_ticks = range(self.r0, self.r0 + (self.nrange+1) * self.dr, self.dr)
        self.theta_ticks = [self.theta0 + self.dtheta * b for b in range(self.nbeam+1)]
        rlabels = [""] * len(self.r_ticks)
        for i in range(0, len(rlabels), 5):
            rlabels[i] = i
        plt.rgrids(self.r_ticks, rlabels)
        plt.thetagrids(self.theta_ticks, range(self.nbeam))


    def _scale_plot(self):
        # Scale min-max
        self.ax.set_thetamin(self.theta_ticks[0])
        self.ax.set_thetamax(self.theta_ticks[-1])
        self.ax.set_rmin(0)
        self.ax.set_rmax(self.r_ticks[-1])


    def plot(self, beams, gates, color="blue"):
        for beam, gate in zip(beams, gates):
            theta = (self.theta0 + beam * self.dtheta) * np.pi / 180        # radians
            r = (self.r0 + gate * self.dr)                                  # km
            width = self.dtheta * np.pi / 180                               # radians
            height = self.dr                                                # km

            x1, x2 = theta, theta + width
            y1, y2 = r, r + height
            x = x1, x2, x2, x1
            y = y1, y1, y2, y2

            self.ax.fill(x, y, color=color)
        self._scale_plot()


if __name__ == '__main__':
    fanplot = FanPlot()
    fanplot.plot([4, 5, 6], [70, 69, 71], "red")
    fanplot.plot([7, 8, 9], [70, 69, 71], "blue")
    fanplot.plot([7, 8, 9], [30, 30, 30], "yellow")

    plt.show()
