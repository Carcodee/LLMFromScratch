import torch as torch
import numpy as np

rng = np.random.default_rng(32);

weights_layer_1 = torch.tensor(rng.uniform(-1.0, 1.0, [4, 4]))

input_matrix = torch.tensor(rng.uniform(-1.0, 1.0, [4, 1]))

biases = torch.tensor(rng.uniform(-1.0, 1.0, [4, 1]))

#print(weights_layer_1)
#print(input_matrix)

output = weights_layer_1 @ input_matrix + biases
print(torch.relu(output))
