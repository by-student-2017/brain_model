import numpy as np
from brain_region_base import BrainRegion

class Amygdala(BrainRegion):
    def process(self, input_signal, neurotransmitters, internal_state=None):
        norepinephrine = neurotransmitters.get('norepinephrine', 1.0)
        serotonin = neurotransmitters.get('serotonin', 1.0)
        signal_strength = np.max(input_signal) * norepinephrine * serotonin
        return signal_strength if signal_strength > 0.3 else 0