import numpy as np
from brain_region_base import BrainRegion

class PrefrontalCortex(BrainRegion):
    def process(self, input_signal, neurotransmitters, internal_state=None):
        dopamine = neurotransmitters.get('dopamine', 1.0)
        glutamate = neurotransmitters.get('glutamate', 1.0)
        weighted_input = np.array(input_signal) * dopamine * glutamate
        exp_input = np.exp(weighted_input)
        return exp_input / np.sum(exp_input)