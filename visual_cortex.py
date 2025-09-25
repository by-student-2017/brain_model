import numpy as np
from brain_region_base import BrainRegion

class VisualCortex(BrainRegion):
    def process(self, image_signal, neurotransmitters, internal_state=None):
        return np.mean(image_signal) * neurotransmitters.get('glutamate', 1.0)