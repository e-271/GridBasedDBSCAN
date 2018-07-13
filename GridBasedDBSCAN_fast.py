"""
Grid-based DBSCAN
Author: Esther Robb

This is the fast implementation of Grid-based DBSCAN.
It uses a sparse Boolean array of data of size (num_grids) x (num_beams)
The data structure is why it is able to run faster - instead of checking all points to
find neighbors, it only checks adjacent points.

Complete implementation.
Confirmed to give the same output as GridBasedDBSCAN_simple.py.
"""

import numpy as np

UNCLASSIFIED = False
NOISE = -1


class GridBasedDBSCAN():

    def __init__(self, f, g, pts_ratio, ngate, nbeam, dr, dtheta, r_init=0):
        dtheta = dtheta * np.pi / 180.0
        self.C = np.zeros((ngate, nbeam))
        for gate in range(ngate):
            for beam in range(nbeam):
                # This is the ratio between radial and angular distance for each point. Across a row it's all the same, consider removing j.
                self.C[gate, beam] = self._calculate_ratio(dr, dtheta, gate, beam, r_init=r_init)
        self.g = g
        self.f = f
        self.pts_ratio = pts_ratio
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
        hgt = self.g        #TODO should there be some rounding happening to accomidate discrete gate/wid?
        wid = self.g / (self.f * self.C[grid_id[0], grid_id[1]])
        ciel_hgt = int(np.ceil(hgt))
        ciel_wid = int(np.ceil(wid))

        # Check for neighbors in a box of shape ciel(2*wid), ciel(2*hgt) around the point
        g_min, g_max = max(0, grid_id[0] - ciel_hgt), min(self.ngate, grid_id[0] + ciel_hgt + 1)
        b_min, b_max = max(0, grid_id[1] - ciel_wid), min(self.nbeam, grid_id[1] + ciel_wid + 1)
        possible_pts = 0
        for g in range(g_min, g_max):
            for b in range(b_min, b_max):
                new_id = (g, b)
                # Add the new point only if it falls within the ellipse defined by wid, hgt
                if self._in_ellipse(new_id, grid_id, hgt, wid):
                    possible_pts += 1
                    if m[new_id]:   # Add the point to seeds only if there is a 1 in the sparse matrix there
                        seeds.append(new_id)
        return seeds, possible_pts


    def _in_ellipse(self, p, q, hgt, wid):
        return ((q[0] - p[0])**2.0 / hgt**2.0 + (q[1] - p[1])**2.0 / wid**2.0) <= 1.0


    def _expand_cluster(self, m, classifications, grid_id, cluster_id, min_points):
        seeds, possible_pts = self._region_query(m, grid_id)
        k = possible_pts * self.pts_ratio
        if len(seeds) < k:
            classifications[grid_id] = NOISE
            return False
        else:
            classifications[grid_id] = cluster_id
            for seed_id in seeds:
                classifications[seed_id] = cluster_id

            while len(seeds) > 0:
                current_point = seeds[0]
                results, possible_pts = self._region_query(m, current_point)
                k = possible_pts * self.pts_ratio
                if len(results) >= k:
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
        m_i - indices where data can be found in the sparse matrix

        Outputs:
        An array with either a cluster id number or dbscan.NOISE (-1) for each
        column vector in m.
        """
        self.m = m
        self.m_i = m_i

        cluster_id = 1
        classifications = np.zeros(m.shape).astype(int) #TODO sparsify
        classifications[:, :] = UNCLASSIFIED

        for grid_id in m_i:
            # Adaptively change one of the epsilon values and the min_points parameter using the C matrix
            if classifications[grid_id] == UNCLASSIFIED:
                if self._expand_cluster(m, classifications, grid_id, cluster_id, self.pts_ratio):
                    cluster_id = cluster_id + 1

        point_labels = [classifications[grid_id] for grid_id in m_i]
        return point_labels


if __name__ == '__main__':
    ngate = 75
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
    data = pickle.load(open("./sas_2018-02-07_grid.pickle", 'rb'))
    gate = data[0][0,:].astype(int)
    beam = data[0][1,:].astype(int)

    from scipy import sparse
    data = sparse.csr_matrix((np.array([True]*len(gate)), (gate, beam)), shape=(ngate, nbeam))
    data_indices = list(zip(gate, beam))

    gdb = GridBasedDBSCAN(f, g, pts_ratio, ngate, nbeam, dr, dtheta, r_init)
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
