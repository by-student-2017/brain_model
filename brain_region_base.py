class BrainRegion:
    def __init__(self, name):
        self.name = name

    def process(self, input_signal, neurotransmitters, internal_state=None):
        raise NotImplementedError