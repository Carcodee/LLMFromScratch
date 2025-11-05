import torch as torch
import numpy as np

rng = np.random.default_rng()

class Layer:
    def __init__(self, neuron_count, layer_minus_one_size, activation_type = "sigmoid"):
        self.neurons = torch.zeros(neuron_count, 1)
        self.bias = torch.zeros(neuron_count, 1)
        self.weights = torch.rand(neuron_count, layer_minus_one_size)
        self.weight_gradients = torch.rand(neuron_count, layer_minus_one_size)
        self.bias_gradients = torch.zeros(neuron_count, 1)
        self.activation_type= activation_type

#inference
    @classmethod
    def load_weights_and_bias(self, neuron_count, bias, weights, layer_minus_one_size, activation_type = "sigmoid"):
        self.neurons = torch.zeros(neuron_count, 1)
        self.bias = bias
        self.weights = weights 
        self.weight_gradients = torch.rand(neuron_count, layer_minus_one_size)
        self.bias_gradients = torch.zeros(neuron_count, 1)
        self.activation_type= activation_type

    def forward(self, neurons_in):
        z = self.weights @ neurons_in + self.bias
        if self.activation_type == "relu":
            self.neurons = torch.relu(z)
        elif self.activation_type == "sigmoid":
            self.neurons = torch.sigmoid(z)
        else:
            self.neurons = z  # linear
        return self.neurons

    def get_derivative(activation_type): 
        pass

class Perceptron():

    def __init__(self):
        self.layers = []
        self.input = None
        self.output_err = None

    def add_layer(self, neuron_count):
        layer = Layer(
            neuron_count= neuron_count, 
            layer_minus_one_size=self.input.shape[0] if len(self.layers) == 0 else self.layers[-1].neurons.shape[0])
        self.layers.append(layer)
        pass

    def set_input(self, input_x, input_y):
        self.input = input_x
        self.input_y = input_y
    
    def forward(self):
        layer_input= self.input 
        for layer in self.layers:
            layer.forward(layer_input)
            layer_input = layer.neurons
            
        self.output_err = torch.sum((self.input_y - self.layers[-1].neurons) ** 2)

    def load_perceptron(self):
        pass


batch_size = 8
data_x = torch.rand(batch_size, 4, 1)
data_y = torch.rand(batch_size, 4, 1)

perceptron = Perceptron()
perceptron.set_input(data_x[0], data_y[0])
perceptron.add_layer(4)
perceptron.add_layer(5)
perceptron.add_layer(8)
perceptron.add_layer(4)

perceptron.forward()
print(perceptron.output_err)
