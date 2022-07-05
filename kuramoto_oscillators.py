import numpy as np
from scipy.integrate import odeint

class Kuramoto:

    def __init__(self, n, timeseries, k, omega_n, theta_n, adjacency_nxn):
        self.n = n
        self.timeseries = timeseries
        self.k = k
        self.omega_n = omega_n
        self.theta_n = theta_n
        self.adjacency_nxn = adjacency_nxn
        self.normalised_coupling = k / n
    
    def derivative(self, theta_n, t=None):
        theta_n1, theta_n2 = np.meshgrid(theta_n, theta_n)
        phase_diff = theta_n2 - theta_n1
        d_theta_n = self.omega_n + self.normalised_coupling * np.sum(self.adjacency_nxn * np.sin(phase_diff), axis = 0)
        return d_theta_n
    
    def phase_timeseries(self):
        theta_nxt = odeint(self.derivative, self.theta_n, self.timeseries)
        return theta_nxt
    
    def coherence_timeseries(self):
        theta_nxt = self.phase_timeseries()
        average_x_t = np.mean(np.cos(theta_nxt), axis = 1)
        average_y_t = np.mean(np.sin(theta_nxt), axis = 1)
        coherence_t = (average_x_t ** 2 + average_y_t ** 2) ** 0.5
        return coherence_t
    
    def mean_frequency_timeseries(self):
        theta_nxt = self.phase_timeseries()
        derivative_t = np.apply_along_axis(self.derivative, axis = 1, arr = theta_nxt)
        omega_avg_t= np.mean(derivative_t, axis = 1)
        return omega_avg_t