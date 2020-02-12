class Connection:
    def __init__(self, neuron_from, weight):
        self.neuron_from = neuron_from
        self.weight = weight
        self.value : float = None