import numpy as np
from brain_region_base import BrainRegion

class Cerebellum(BrainRegion):
    def process(self, input_signal, neurotransmitters, internal_state=None):
        gaba = neurotransmitters.get('gaba', 1.0)
        glutamate = neurotransmitters.get('glutamate', 1.0)
        return np.array(input_signal) * gaba * glutamate