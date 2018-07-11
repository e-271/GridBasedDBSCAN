"""
Grid-based DBSCAN
Author: Esther Robb

This is the fast implementation of Grid-based DBSCAN.
It uses a sparse Boolean array of data of size (num_grids) x (num_beams)
The data structure is why it is able to run faster - instead of checking all points to
find neighbors, it only checks adjacent points.
"""

import numpy as np

UNCLASSIFIED = False
NOISE = -1


class GridBasedDBSCAN():

    def __init__(self, gate_eps, beam_eps, time_eps, min_pts, ngate, nbeam, dr, dtheta, r_init=0):
        dtheta = dtheta * np.pi / 180.0
        self.C = np.zeros((ngate, nbeam))
        for g in range(ngate):
            for b in range(nbeam):
                # This is the ratio between radial and angular distance for each point. Across a row it's all the same, consider removing j.
                self.C[g,b] = self._calculate_ratio(dr, dtheta, g, b, r_init=r_init)
        self.gate_eps = gate_eps
        self.beam_eps = beam_eps
        self.time_eps = time_eps
        self.min_pts = min_pts
        self.ngate = ngate
        self.nbeam = nbeam


    # Input for grid-based DBSCAN:
    # C matrix calculated based on sensor data.
    # There is very little variance from beam to beam for our radars - down to the 1e-16 level.
    def _calculate_ratio(self, dr, dt, i, j, r_init=0):
        r_init, dr, dt, i, j = float(r_init), float(dr), float(dt), float(i), float(j)
        cij = (r_init + dr * i) / (2.0 * dr) * (np.sin(dt * (j + 1.0) - dt * j) + np.sin(dt * j - dt * (j - 1.0)))
        return cij


    def _region_query(self, m, grid_id):
        seeds = []
        hgt = self.gate_eps        #TODO should there be some rounding happening to accomidate discrete gate/wid?
        wid = self.beam_eps / self.C[grid_id[0], grid_id[1]]
        ciel_hgt = int(np.ceil(hgt))
        ciel_wid = int(np.ceil(wid))

        # Check for neighbors in a box of shape ciel(2*wid), ciel(2*hgt) around the point
        g_min, g_max = max(0, grid_id[0] - ciel_hgt), min(self.ngate, grid_id[0] + ciel_hgt + 1)
        b_min, b_max = max(0, grid_id[1] - ciel_wid), min(self.nbeam, grid_id[1] + ciel_wid + 1)
        for g in range(g_min, g_max):
            for b in range(b_min, b_max):
                new_id = (g, b)
                # Skip ahead if no point is found at this index
                if not m[new_id]:
                    continue
                # Add the new point only if it falls within the ellipse defined by wid, hgt
                if self._in_ellipse(new_id, grid_id, hgt, wid):
                    seeds.append(new_id)

        return seeds


    def _in_ellipse(self, p, q, hgt, wid):
        return ((q[0] - p[0])**2.0 / hgt**2.0 + (q[1] - p[1])**2.0 / wid**2.0) <= 1.0


    def _expand_cluster(self, m, classifications, grid_id, cluster_id, min_points):
        seeds = self._region_query(m, grid_id)
        if len(seeds) < min_points:
            classifications[grid_id] = NOISE
            return False
        else:
            classifications[grid_id] = cluster_id
            for seed_id in seeds:
                classifications[seed_id] = cluster_id

            while len(seeds) > 0:
                current_point = seeds[0]
                results = self._region_query(m, current_point)
                eps = self.gate_eps, self.beam_eps / self.C[current_point[0], current_point[1]]
                if len(results) >= min_points:
                    for i in range(0, len(results)):
                        result_point = results[i]
                        if classifications[result_point] == UNCLASSIFIED or classifications[result_point] == NOISE:
                            if classifications[result_point] == UNCLASSIFIED:
                                seeds.append(result_point)
                            classifications[result_point] = cluster_id
                seeds = seeds[1:]
            return True


    def fit(self, m, m_i):
        """
        Inputs:
        m - A csr_sparse bool matrix, num_gates x num_beams x num_times
        eps - Maximum distance two points can be to be regionally related
        min_points - The minimum number of points to make a cluster

        Outputs:
        An array with either a cluster id number or dbscan.NOISE (-1) for each
        column vector in m.
        """
        self.m = m
        self.m_i = m_i

        g, f = self.beam_eps, 1
        cluster_id = 1
        n_points = len(m_i)
        classifications = np.zeros(m.shape).astype(int) #TODO sparsify
        classifications[:, :] = UNCLASSIFIED

        for grid_id in m_i:
            # Adaptively change one of the epsilon values and the min_points parameter using the C matrix
            if classifications[grid_id] == UNCLASSIFIED:
                if self._expand_cluster(m, classifications, grid_id, cluster_id, self.min_pts):
                    cluster_id = cluster_id + 1

        point_labels = [classifications[grid_id] for grid_id in m_i]
        return point_labels


if __name__ == '__main__':
    from scipy import sparse
    ngate = 75
    nbeam = 16
    pts_per_time = 78
    np.random.seed(2)

    beam = np.random.randint(low=0, high=nbeam - 1, size=pts_per_time)
    gate = np.random.randint(low=0, high=ngate - 1, size=pts_per_time)
    data = sparse.csr_matrix((np.array([True]*pts_per_time), (gate, beam)), shape=(ngate, nbeam))
    data_indices = list(zip(gate, beam))

    gate_eps = 5
    beam_eps = 5
    time_eps = 10
    min_pts = 4

    dr = 45
    dtheta = 3.3
    r_init = 180

    gdb = GridBasedDBSCAN(gate_eps, beam_eps, time_eps, min_pts, ngate, nbeam, dr, dtheta, r_init)
    labels = gdb.fit(data, data_indices)

    from FanPlot import FanPlot
    import matplotlib.pyplot as plt
    clusters = np.unique(labels)
    print('Clusters: ', clusters)
    colors = list(plt.cm.plasma(np.linspace(0, 1, len(clusters))))  # one extra unused color at index 0 (no cluster label == 0)
    colors.append((0, 0, 0, 1)) # black for noise
    fanplot = FanPlot(nrange=ngate, nbeam=nbeam)
    for c in clusters:
        label_mask = labels == c
        fanplot.plot(beam[label_mask], gate[label_mask], colors[c])
    plt.show()
