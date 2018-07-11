"""
Grid-based DBSCAN
Author: Esther Robb

This is the simple/slower implementation of Grid-based DBSCAN.

It uses an array of radar data points in this format:
[beam_number, gate_number] e.g., [14, 75]

as opposed to the grid of size [num_beams, num_gates] filled with 0's and 1's used by the faster implementation.
"""

import numpy as np

UNCLASSIFIED = False
NOISE = -1


class GridBasedDBSCAN():


    def __init__(self, gate_eps, beam_eps, min_pts, nrang, nbeam, dr, dtheta, r_init=0):
        dtheta = dtheta * np.pi / 180.0
        self.C = np.zeros((nrang, nbeam))
        for i in range(nrang):
            for j in range(nbeam):
                # This is the ratio between radial and angular distance for each point. Across a row it's all the same, consider removing j.
                self.C[i,j] = self._calculate_ratio(dr, dtheta, i, j, r_init=r_init)
        self.gate_eps = gate_eps
        self.beam_eps = beam_eps
        self.min_pts = min_pts


    def _eps_neighborhood(self, p, q, space_eps):
        h = space_eps[0]
        w = space_eps[1]
        # Search in an ellipse with widths defined by the 2 epsilon values
        in_ellipse = ((q[0] - p[0])**2 / h**2 + (q[1] - p[1])**2 / w**2) <= 1
        return in_ellipse


    def _region_query(self, m, point_id):
        n_points = m.shape[1]
        seeds = []
        gate, beam = m[0, point_id], m[1, point_id]
        eps = (self.gate_eps, self.beam_eps / self.C[gate, beam])
        for i in range(0, n_points):
            if self._eps_neighborhood(m[:, point_id], m[:, i], eps):
                seeds.append(i)
        return seeds


    def _expand_cluster(self, m, classifications, point_id, cluster_id, min_points):
        seeds = self._region_query(m, point_id)
        if len(seeds) < min_points:
            classifications[point_id] = NOISE
            return False
        else:
            classifications[point_id] = cluster_id
            for seed_id in seeds:
                classifications[seed_id] = cluster_id

            while len(seeds) > 0:
                current_point = seeds[0]
                #eps = (self.gate_eps, self.beam_eps / C[current_point[0], current_point[1]])
                results = self._region_query(m, current_point)
                if len(results) >= min_points:
                    for i in range(0, len(results)):
                        result_point = results[i]
                        if classifications[result_point] == UNCLASSIFIED or \
                                classifications[result_point] == NOISE:
                            if classifications[result_point] == UNCLASSIFIED:
                                seeds.append(result_point)
                            classifications[result_point] = cluster_id
                seeds = seeds[1:]
            return True


    # Input for grid-based DBSCAN:
    # C matrix calculated based on sensor data.
    def _calculate_ratio(self, dr, dt, i, j, r_init=0):
        r_init, dr, dt, i, j = float(r_init), float(dr), float(dt), float(i), float(j)
        cij = (r_init + dr * i) / (2.0 * dr) * (np.sin(dt * (j + 1.0) - dt * j) + np.sin(dt * j - dt * (j - 1.0)))
        return cij


    def fit(self, m):
        """
        Inputs:
        m - A matrix whose rows are [gate, beam, time]
        eps - Maximum distance two points can be to be regionally related
        min_points - The minimum number of points to make a cluster

        Outputs:
        An array with either a cluster id number or dbscan.NOISE (-1) for each
        column vector in m.
        """
        g, f = self.beam_eps, 1
        cluster_id = 1
        n_points = m.shape[1]
        classifications = [UNCLASSIFIED] * n_points
        for point_id in range(0, n_points):
            point = m[:, point_id]
            i, j = int(point[0]), int(point[1]) # range gate, beam
            wid = g / (f * self.C[i, j])
            # Adaptively change one of the epsilon values and the min_points parameter using the C matrix
            if classifications[point_id] == UNCLASSIFIED:
                if self._expand_cluster(m, classifications, point_id, cluster_id, self.min_pts):
                    cluster_id = cluster_id + 1
        return classifications


if __name__ == '__main__':
    nrang = 75
    nbeam = 16
    gate = np.random.randint(low=0, high=nrang-1, size=100)
    beam = np.random.randint(low=0, high=nbeam-1, size=100)
    gate_eps = 5
    beam_eps = 5
    min_pts = 4

    dr = 45
    dtheta = 3.3
    r_init = 180

    data = np.row_stack((gate, beam))
    gdb = GridBasedDBSCAN(gate_eps, beam_eps, min_pts, nrang, nbeam, dr, dtheta, r_init)
    labels = gdb.fit(data)

    from FanPlot import FanPlot
    import matplotlib.pyplot as plt
    clusters = np.unique(labels)
    print('Clusters: ', clusters)
    colors = list(plt.cm.plasma(np.linspace(0, 1, len(clusters))))  # one extra unused color at index 0 (no cluster label == 0)
    colors.append((0, 0, 0, 1)) # black for noise
    fanplot = FanPlot()
    for c in clusters:
        label_mask = labels == c
        fanplot.plot(beam[label_mask], gate[label_mask], colors[c])
    plt.show()

