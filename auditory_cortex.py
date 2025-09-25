import numpy as np
from brain_region_base import BrainRegion

class AuditoryCortex(BrainRegion):
    def process(self, auditory_input, neurotransmitters, internal_state=None):
        return np.mean(auditory_input) * neurotransmitters.get('serotonin', 1.0)