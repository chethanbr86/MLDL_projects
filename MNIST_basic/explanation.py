import numpy as np

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] #sizes[1:] skips the input layer (since it doesn't have biases) and includes all subsequent layers.
        self.weights = [np.random.randn(z, x) for x, z in zip(sizes[:-1], sizes[1:])] #zip(sizes[:-1], sizes[1:]) pairs each layer with the next one, creating tuples (x, z) where:
        # x is the number of neurons in the current layer.
        # z is the number of neurons in the next layer.
        print(f'sizes: {sizes}\n, biases: {self.biases}\n, weights: {self.weights}\n')

    def feedforward(self, a):
        # Return the output of the network if "a" is input.
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

net = Network([2, 3, 1]) #like[784,30,10]
net1 = Network.feedforward(5)
print(net)
print(net1)