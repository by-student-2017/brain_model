import numpy as np
from brain_region_base import BrainRegion

class Hippocampus(BrainRegion):
    def __init__(self, name):
        super().__init__(name)
        self.memory_buffer = []

    def process(self, input_signal, neurotransmitters, internal_state=None):
        glutamate = neurotransmitters.get('glutamate', 1.0)
        acetylcholine = neurotransmitters.get('acetylcholine', 1.0)
        memory_trace = np.array(input_signal) * glutamate * acetylcholine
        self.memory_buffer.append(memory_trace)
        return memory_trace