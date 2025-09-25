import numpy as np
from brain_region_base import BrainRegion

class Striatum(BrainRegion):
    def process(self, input_signal, neurotransmitters, internal_state=None):
        lambda_ = neurotransmitters.get('dopamine_decay', 0.5)
        return np.exp(-lambda_ * np.array(input_signal))