import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.dates import num2date

#TODO slate to add to utilities

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


    def _monotonically_increasing(self, vec):
        if len(vec) < 2:
            return True
        return all(x <= y for x, y in zip(vec[:-1], vec[1:]))


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


    def plot_all(self, times_unique_dt, times_unique_num, times_num, beams, gates, labels, colors, base_path=""):
        scan = 0
        i = 0
        plt.close(self.fig)
        unique_labels = np.unique(labels)
        while i < len(times_unique_num):
            j = 0
            while i + j + 1 <= len(times_unique_num):
                new_scan_mask = (
                        (times_num >= times_unique_num[i]).astype(int) &
                        (times_num <= times_unique_num[i + j]).astype(int)
                ).astype(bool)
                if self._monotonically_increasing(beams[new_scan_mask]):
                    scan_mask = new_scan_mask
                    j += 1
                else:
                    break
            beams_i = beams[scan_mask]
            gates_i = gates[scan_mask]
            self._open_figure()
            self._scale_plot()
            for c, label in enumerate(unique_labels):
                label_mask = labels[scan_mask] == label
                self.plot(beams_i[label_mask], gates_i[label_mask], color=colors[label])

            # plt.show()
            scan += 1
            plt.title(str(times_unique_dt[i]))
            plt.savefig(base_path + "fanplot" + str(scan) + ".png")
            plt.close()
            i += j

if __name__ == '__main__':
    fanplot = FanPlot()
    fanplot.plot([4, 5, 6], [70, 69, 71], "red")
    fanplot.plot([7, 8, 9], [70, 69, 71], "blue")
    fanplot.plot([7, 8, 9], [30, 30, 30], "yellow")

    plt.show()
