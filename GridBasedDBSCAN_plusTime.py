"""
Grid-based DBSCAN + Time filter
Author: Esther Robb

This is the simple/slower implementation of Grid-based DBSCAN.
It uses an array of radar data points in this format:
[beam_number, gate_number, time_secs]
e.g.,
[14, 75, 3.5]

The time filter concept was based on Brant & Kut 2006 "ST-DBSCAN".
The search area will be an ellipse in the spatial dimensions and an elliptic cylinder in the space+time dimensions
If you want an ellipsoid search area in space+time dimensions, uncomment the corresponding code in _eps_neighborhood
"""

import numpy as np

UNCLASSIFIED = False
NOISE = -1


class GridBasedDBSCAN():

    def __init__(self, gate_eps, beam_eps, time_eps, min_pts, nrang, nbeam, dr, dtheta, r_init=0):
        dtheta = dtheta * np.pi / 180.0
        self.C = np.zeros((nrang, nbeam))
        for i in range(nrang):
            for j in range(nbeam):
                # This is the ratio between radial and angular distance for each point. Across a row it's all the same, consider removing j.
                self.C[i,j] = self._calculate_ratio(dr, dtheta, i, j, r_init=r_init)
        self.gate_eps = gate_eps
        self.beam_eps = beam_eps
        self.time_eps = time_eps
        self.min_pts = min_pts


    def _eps_neighborhood(self, p, q, space_eps):
        # Filter by time neighbors
        min_time = p[2] - self.time_eps
        max_time = p[2] + self.time_eps
        time_neighbor = q[2] >= min_time and q[2] <= max_time
        if not time_neighbor:
            return False

        h = space_eps[0]
        w = space_eps[1]

        # Search in an ellipsoid with the 3 epsilon values (slower, results are similar but not the same)
        # t = self.time_eps
        # in_ellipse = ((q[0] - p[0])**2 / w**2 + (q[1] - p[1])**2 / h**2 + (q[2] - p[2])**2 / t**2) <= 1

        # Search in an ellipse with widths defined by the 2 epsilon values
        in_ellipse = ((q[0] - p[0])**2 / h**2 + (q[1] - p[1])**2 / w**2) <= 1

        return in_ellipse


    def _region_query(self, m, point_id, eps):
        n_points = m.shape[1]
        seeds = []

        for i in range(0, n_points):
            if self._eps_neighborhood(m[:, point_id], m[:, i], eps):
                seeds.append(i)
        return seeds


    def _expand_cluster(self, m, classifications, point_id, cluster_id, eps, min_points):
        seeds = self._region_query(m, point_id, eps)
        if len(seeds) < min_points:
            classifications[point_id] = NOISE
            return False
        else:
            classifications[point_id] = cluster_id
            for seed_id in seeds:
                classifications[seed_id] = cluster_id

            while len(seeds) > 0:
                current_point = seeds[0]
                results = self._region_query(m, current_point, eps)
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
    # Based on Birant et al
    # TODO why does this make a matrix if it doesn't vary from beam to beam
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
            eps = (self.gate_eps, wid)
            # Adaptively change one of the epsilon values and the min_points parameter using the C matrix
            if classifications[point_id] == UNCLASSIFIED:
                if self._expand_cluster(m, classifications, point_id, cluster_id, eps, self.min_pts):
                    cluster_id = cluster_id + 1
        return classifications
