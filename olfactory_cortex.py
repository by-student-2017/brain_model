from brain_region_base import BrainRegion
import numpy as np

class OlfactoryCortex(BrainRegion):
    def process(self, olfactory_input, neurotransmitters, internal_state):
        # The stronger the olfactory stimulus, the more unpleasant it is (e.g., moldy smell).
        intensity = np.array(olfactory_input)
        discomfort = intensity * (1 - neurotransmitters.get("serotonin", 0.5))
        return discomfort
