# insula.py

"""
Insula class models the neural mechanisms of empathy-based altruistic behavior.
It processes observed pain signals from others, integrates internal bodily states,
and modulates emotional output toward the amygdala. This class is designed for
future extensibility, including reinforcement learning, social context integration,
and time-series tracking of empathy responses.

Key Features:
- Empathy signal processing based on observed pain
- Internal state integration (e.g., own pain, stress)
- Modulation of empathy strength and output to amygdala
- Short-term and long-term empathy memory
- Extensible architecture for reinforcement learning and social context modeling
"""

import numpy as np
from brain_region_base import BrainRegion

class Insula(BrainRegion):
    def __init__(self, name="Insula"):
        super().__init__(name)
        self.empathy_weights = {
            'pain': 1.0,
            'distress': 1.0,
            'fear': 1.0,
            'sadness': 1.0
        }
        self.short_term_empathy = {}  # STP-like temporary empathy memory
        self.long_term_empathy = {}   # LTP-like persistent empathy memory
        self.stp_decay_rate = 0.9     # Short-term empathy decay rate
        self.ltp_learning_rate = 0.01 # Long-term empathy learning rate

    def update_empathy_memory(self, signal_type, signal_strength):
        # Short-term potentiation
        self.short_term_empathy[signal_type] = signal_strength

        # Long-term potentiation
        prev = self.long_term_empathy.get(signal_type, 1.0)
        self.long_term_empathy[signal_type] = prev + self.ltp_learning_rate * (signal_strength - prev)

    def decay_stp(self):
        for signal_type in self.short_term_empathy:
            self.short_term_empathy[signal_type] *= self.stp_decay_rate

    def process(self, observed_pain_signal, internal_state=None, social_context=None):
        """
        Process observed pain signal and internal state to generate empathy output.
        Parameters:
            observed_pain_signal (dict): e.g., {'pain': 0.8, 'distress': 0.6}
            internal_state (dict): e.g., {'own_pain': 0.5, 'stress': 0.3}
            social_context (dict): optional future extension
        Returns:
            empathy_output (float): strength of empathy signal to amygdala
        """
        if internal_state is None:
            internal_state = {}

        # Combine observed pain and internal state
        empathy_signal = 0.0
        for signal_type, strength in observed_pain_signal.items():
            internal_modulation = internal_state.get('own_' + signal_type, 1.0)
            weight = self.empathy_weights.get(signal_type, 1.0)
            combined_strength = strength * internal_modulation * weight
            empathy_signal += combined_strength

            # Update memory
            self.update_empathy_memory(signal_type, combined_strength)

        # Normalize output
        empathy_output = np.tanh(empathy_signal)

        return empathy_output

    # Future extension points:
    # - Incorporate reinforcement learning to adjust empathy_weights
    # - Use time-series tracking of empathy responses for adaptive behavior
    # - Integrate social context (e.g., group membership, familiarity)
    # - Connect to amygdala and prefrontal cortex for action modulation