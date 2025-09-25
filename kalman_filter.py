import numpy as np

class KalmanFilter:
    def __init__(self, initial_state, initial_uncertainty, process_variance, observation_variance):
        self.state_estimate = initial_state
        self.uncertainty = initial_uncertainty
        self.process_variance = process_variance
        self.observation_variance = observation_variance

    def update(self, observation):
        predicted_state = self.state_estimate
        predicted_uncertainty = self.uncertainty + self.process_variance

        kalman_gain = predicted_uncertainty / (predicted_uncertainty + self.observation_variance)
        self.state_estimate = predicted_state + kalman_gain * (observation - predicted_state)
        self.uncertainty = (1 - kalman_gain) * predicted_uncertainty

        return self.state_estimate