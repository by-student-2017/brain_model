import numpy as np
from brain_region_base import BrainRegion

class Brainstem(BrainRegion):
    def process(self, input_signal, neurotransmitters, internal_state):
        serotonin = neurotransmitters.get('serotonin', 1.0)
        norepinephrine = neurotransmitters.get('norepinephrine', 1.0)
        heart_rate = internal_state.get('heart_rate', 70)
        return np.mean(input_signal) * serotonin * norepinephrine * (heart_rate / 70)