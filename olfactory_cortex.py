from brain_region_base import BrainRegion
import numpy as np

class OlfactoryCortex(BrainRegion):
    def __init__(self, name):
        super().__init__(name)
        self.previous_intensity = 0.0
        self.desensitization_rate = 0.1  # Sensitivity reduction rate

    def process(self, olfactory_input, neurotransmitters, internal_state):
        intensity = np.array(olfactory_input)[0]
        delta = abs(intensity - self.previous_intensity)

        # If stimulation continues, gradually reduce sensitivity
        if delta < 0.05:
            intensity *= (1 - self.desensitization_rate)

        self.previous_intensity = intensity

        serotonin_level = neurotransmitters.get("serotonin", 0.5)
        discomfort = intensity * (1 - serotonin_level)

        return [discomfort]