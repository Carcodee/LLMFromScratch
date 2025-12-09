import torch

class Hidden(torch.nn.Module):
    def __init__(self, input_size, neuron_count, activation = "none"):
        torch.manual_seed(0)
        self.W = torch.randn([neuron_count, input_size])
        #broadcast will expand 1 to match batch size
        self.b = torch.randn([1, neuron_count])
        self.W_grad = torch.randn([input_size, neuron_count])
        self.b_grad = torch.randn([1, neuron_count])
        self.activation = activation
        pass
    def forward(self, input):
        self.z = input @ self.W.T + self.b
        self.deltas = torch.zeros(self.z.shape)
        self.activation = self.apply_activation(self.z)
        return self.activation

    def calculate_grad(self, next_layers_weights, next_layer_deltas, y_exp = None):
        if y is not None:
            return 2 * (self.y - y_exp)
        self.deltas = next_layer_deltas @ next_layers_weights.T  
        self.deltas = self.deltas * self.W
        self.W_grad = self.W * self.deltas
        pass

    def apply_activation(self, z): 
        if self.activation == "relu":
            return torch.relu(z)
        elif self.activation == "sigmoid":
            self.neurons = torch.sigmoid(z)
        else:
            return z  #linear

    def get_activations_dev(self):
        if self.activation == "relu":
            return (self.y > 0).float()
        elif self.activation == "sigmoid":
            return self.neurons * (1 - self.neurons)
        else:
            return 1

class MLP():
    def __init__(self):
        self.layers = []
        pass
    def add_hidden(self, size_in, neuron_count):
        self.layers.append(Hidden(size_in, neuron_count))
    def forward(self, train_x):
        input = train_x
        for hidden in self.layers:
            hidden.forward(input)
            input = hidden.activation
    def calculate_deltas(self):
        for n in (len(self.layers - 1), 0, -1):

            pass




#x
#shape in dim 0 = we have batch count
#shape in dim 1 = we have input count

#win dim 0 we have batch count in
#win dim 1 neuron count out 

x = torch.ones([2, 3])

#print(x)
#print(w)

torch.manual_seed(0)
w = torch.randn([3, 4])
b = torch.randn([1, 4])
y = x @ w + b
print(y)

mlp = MLP()


mlp.add_hidden(3, 4)
mlp.forward(x)
print(mlp.layers[-1].z)