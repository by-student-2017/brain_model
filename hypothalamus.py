import numpy as np
from brain_region_base import BrainRegion

class Hypothalamus(BrainRegion):
    def process(self, input_signal, neurotransmitters, internal_state):
        oxytocin = neurotransmitters.get('oxytocin', 1.0)
        vasopressin = neurotransmitters.get('vasopressin', 1.0)
        temp = internal_state.get('body_temperature', 36.5)
        hormone_signal = np.mean(input_signal) * oxytocin * vasopressin * (temp / 37.0)
        return hormone_signal