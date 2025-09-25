import numpy as np
from brain_region_base import BrainRegion

class Midbrain(BrainRegion):
    def process(self, input_signal, neurotransmitters, internal_state=None):
        ach = neurotransmitters.get('acetylcholine', 1.0)
        dopamine = neurotransmitters.get('dopamine', 1.0)
        return np.mean(input_signal) * ach * dopamine