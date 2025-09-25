import numpy as np
from brain_region_base import BrainRegion

class LanguageArea(BrainRegion):
    def process(self, linguistic_input, neurotransmitters, internal_state=None):
        return np.sum(linguistic_input) * neurotransmitters.get('acetylcholine', 1.0)