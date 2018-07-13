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


    def __init__(self, f, g, k, ngate, nbeam, dr, dtheta, r_init=0):
        dtheta = dtheta * np.pi / 180.0
        self.C = np.zeros((ngate, nbeam))
        for i in range(ngate):
            for j in range(nbeam):
                # This is the ratio between radial and angular distance for each point. Across a row it's all the same, consider removing j.
                self.C[i,j] = self._calculate_ratio(dr, dtheta, i, j, r_init=r_init)
        self.g = g
        self.f = f
        self.k = k
        self.ngate = ngate
        self.nbeam = nbeam


    # Input for grid-based DBSCAN:
    # C matrix calculated based on sensor data.
    def _calculate_ratio(self, dr, dt, i, j, r_init=0):
        r_init, dr, dt, i, j = float(r_init), float(dr), float(dt), float(i), float(j)
        cij = (r_init + dr * i) / (2.0 * dr) * (np.sin(dt * (j + 1.0) - dt * j) + np.sin(dt * j - dt * (j - 1.0)))
        return cij


    def _in_ellipse(self, p, q, g_eps, b_eps):
        # Search in an ellipse with widths defined by the 2 epsilon values
        in_ellipse = ((q[0] - p[0]) ** 2 / g_eps ** 2 + (q[1] - p[1]) ** 2 / b_eps ** 2) <= 1
        return in_ellipse


    # TODO is there a more efficient way to do this?
    def _possible_observations(self, p, gate_eps, beam_eps):
        possible_observations = 0
        ciel_hgt = int(np.ceil(gate_eps))
        ciel_wid = int(np.ceil(beam_eps))
        # Check for possible observations in a box of shape ciel(2*wid), ciel(2*hgt) around the point
        g_min, g_max = max(0, p[0] - ciel_hgt), min(self.ngate, p[0] + ciel_hgt + 1)
        b_min, b_max = max(0, p[1] - ciel_wid), min(self.nbeam, p[1] + ciel_wid + 1)
        for g in range(g_min, g_max):
            for b in range(b_min, b_max):
                q = (g, b)
                if self._in_ellipse(q, p, gate_eps, beam_eps):
                    possible_observations += 1
        return possible_observations


    def _region_query(self, m, point_id):
        n_points = m.shape[1]
        seeds = []
        g, b = int(m[0, point_id]), int(m[1, point_id])
        gate_eps = self.g
        beam_eps = self.g / (self.f * self.C[g, b])
        for i in range(0, n_points):
            if self._in_ellipse(m[:, point_id], m[:, i], gate_eps, beam_eps):
                seeds.append(i)
        min_pts = self._possible_observations((g, b), gate_eps, beam_eps) * self.k
        return seeds, min_pts


    def _expand_cluster(self, m, classifications, point_id, cluster_id):
        seeds, min_pts = self._region_query(m, point_id)
        if len(seeds) < min_pts:
            classifications[point_id] = NOISE
            return False
        else:
            classifications[point_id] = cluster_id
            for seed_id in seeds:
                classifications[seed_id] = cluster_id

            while len(seeds) > 0:
                current_point = seeds[0]
                results, min_pts = self._region_query(m, current_point)
                if len(results) >= min_pts:
                    for i in range(0, len(results)):
                        result_point = results[i]
                        if classifications[result_point] == UNCLASSIFIED or \
                                classifications[result_point] == NOISE:
                            if classifications[result_point] == UNCLASSIFIED:
                                seeds.append(result_point)
                            classifications[result_point] = cluster_id
                seeds = seeds[1:]
            return True


    def fit(self, m):
        """
        Inputs:
        m - A matrix whose rows are [gate, beam]

        Outputs:
        An array with either a cluster id number or dbscan.NOISE (-1) for each
        column vector in m.
        """
        cluster_id = 1
        n_points = m.shape[1]
        classifications = [UNCLASSIFIED] * n_points
        for point_id in range(0, n_points):
            if classifications[point_id] == UNCLASSIFIED:
                if self._expand_cluster(m, classifications, point_id, cluster_id):
                    cluster_id = cluster_id + 1
        return classifications


if __name__ == '__main__':
    nrang = 75
    nbeam = 16
    g = 4               #  (g/f >= 5) prevents (w[i,j] = g / f * C[i,j]) from dropping below 1 with the other settings
    f = 1               #  but the results from dont make me happy
    pts_ratio = 0.3
    dr = 45
    dtheta = 3.3
    r_init = 180


    """ Fake data 
    gate = np.random.randint(low=0, high=nrang-1, size=100)
    beam = np.random.randint(low=0, high=nbeam-1, size=100)
    data = np.row_stack((gate, beam))
    """
    """ Real radar data """
    import pickle
    data = pickle.load(open("sas_2-7-18_scan0.pickle", 'rb'))
    gate = data[0,:]
    beam = data[1,:]

    """ Grid-based DBSCAN """
    gdb = GridBasedDBSCAN(f, g, pts_ratio, nrang, nbeam, dr, dtheta, r_init)
    labels = gdb.fit(data)

    from FanPlot import FanPlot
    import matplotlib.pyplot as plt
    clusters = np.unique(labels)
    print('Grid-based DBSCAN Clusters: ', clusters)
    colors = list(plt.cm.plasma(np.linspace(0, 1, len(clusters))))  # one extra unused color at index 0 (no cluster label == 0)
    colors.append((0, 0, 0, 1)) # black for noise

    # Plot a fanplot
    fanplot = FanPlot()
    for c in clusters:
        label_mask = labels == c
        fanplot.plot(beam[label_mask], gate[label_mask], colors[c])
    plt.title('Grid-based DBSCAN fanplot')
    plt.show()
    for c in clusters:
        label_mask = labels == c
        plt.scatter(beam[label_mask], gate[label_mask], color=colors[c])
    plt.title('Grid-based DBSCAN gridplot (what regular DBSCAN sees)')
    plt.show()


    """ Regular DBSCAN """
    from sklearn.cluster import DBSCAN
    dbs_eps, min_pts = 5, 5
    dbs_data = data.T
    dbscan = DBSCAN(eps=5, min_samples=5)
    labels = dbscan.fit_predict(dbs_data)
    clusters = np.unique(labels)
    print('Regular DBSCAN Clusters: ', clusters)

    colors = list(plt.cm.plasma(np.linspace(0, 1, len(clusters))))  # one extra unused color at index 0 (no cluster label == 0)
    colors.append((0, 0, 0, 1)) # black for noise
    fanplot = FanPlot()
    for c in clusters:
        label_mask = labels == c
        fanplot.plot(beam[label_mask], gate[label_mask], colors[c])
    plt.title('Regular DBSCAN fanplot')
    plt.show()
    for c in clusters:
        label_mask = labels == c
        plt.scatter(beam[label_mask], gate[label_mask], color=colors[c])
    plt.title('Regular DBSCAN gridplot (what regular DBSCAN sees)')
    plt.show()