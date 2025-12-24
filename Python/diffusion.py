#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.io import read_image
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path


# %%
target_img_size = [50, 50]
train_size = 0.9
channels = 1
flatten_size = target_img_size[0] *  target_img_size[1] * channels
latent_space_size = 500 

#torch.manual_seed(329846)
device =  'cuda' if torch.cuda.is_available() else 'cpu'

# %%
images_path = './images_data/human'
p = Path(images_path)
img_tensor_uint8 = torch.stack([TF.resize(read_image(item.resolve()), target_img_size)
                                           for item in p.iterdir()]).to(device=device)
                                

# %%
imgs_tensor = img_tensor_uint8.float().to(device=device) / 255.0

#H,W,C
imgs_tensor = imgs_tensor.transpose(dim0=1, dim1=-1).transpose(dim0=1, dim1=2).to(device=device)
imgs_tensor = imgs_tensor[:, :, :, 3 - channels:]
print(imgs_tensor.shape)
train_size_int = int(imgs_tensor.shape[0] * train_size)

train_images_x = imgs_tensor[:train_size_int].clone().to(device=device)
test_images_x = imgs_tensor[train_size_int:].clone().to(device=device)

torch.manual_seed(123465)
mu = 0
std_train = 0.1
std_test = 0.1
random_nums_train = torch.randn_like(train_images_x).to(device=device)
random_nums_test = torch.randn_like(test_images_x).to(device=device)

train_images_x = train_images_x + random_nums_train * std_train
test_images_x = test_images_x + random_nums_test * std_test

train_images_y = imgs_tensor[:train_size_int].to(device=device)
test_images_y = imgs_tensor[train_size_int:].to(device=device)



#we assume that 0 is the batch dim
def get_batch_random(src_x, src_y, batch_size):
    batches_to_pick = torch.randint(0, src_x.shape[0], (batch_size, )).to(device=device)
    batches_x = src_x[batches_to_pick].to(device=device)
    batches_y = src_y[batches_to_pick].to(device=device)
    batches_x = batches_x.reshape(batch_size, -1).to(device=device)
    batches_y = batches_y.reshape(batch_size, -1).to(device=device)
    return batches_x, batches_y

def get_batches(src_x, src_y, batch_size = -1):
    if batch_size == -1:
        #pick all batches
        batch_size = src_x.shape[0]

    src_sliced_x = src_x[:batch_size].to(device= device)
    src_sliced_y = src_y[:batch_size].to(device= device)
    batches_x = src_sliced_x.reshape(batch_size, -1).to(device= device)
    batches_y = src_sliced_y.reshape(batch_size, -1).to(device= device)
    return batches_x, batches_y

def get_img_from_flatten(img_batch):
    return img_batch.view(-1, target_img_size[0], target_img_size[1], channels)

def flatten_batches(img_batch):
    return img_batch.view(img_batch[0], -1)

def get_random_latent_img(latent_size):
    return torch.rand_like(torch.zeros([1, latent_size])).to(device=device)

# %%
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(flatten_size, latent_space_size * 2),
            torch.nn.Sigmoid(),
            torch.nn.LayerNorm(latent_space_size * 2),
            torch.nn.Linear(latent_space_size * 2, latent_space_size),
        )
    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(latent_space_size, latent_space_size * 2),
            torch.nn.Sigmoid(),
            torch.nn.LayerNorm(latent_space_size * 2),
            torch.nn.Linear(latent_space_size * 2, flatten_size),
        )
    def forward(self, x):
        return self.model(x)

class AutoEncoder(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    def forward(self, x, target = None, isolate = 'none'):
        loss = None
        if isolate == 'encoder':
            x = self.encoder(x)
        elif isolate == 'decoder':
            x = self.decoder(x)
        else:
            x = self.encoder(x)
            x = self.decoder(x)
            if target is not None:
                loss = F.mse_loss(x, target)

        return x, loss


# %%
auto_encoder = AutoEncoder().to(device=device)
optimizer = torch.optim.Adam(auto_encoder.parameters(), lr=1e-3)

# %%
#visulization
activations = {}
def save_activation(name):
    def hook(module, input, output):
        activations[name] = output.detach().cpu()
    return hook

for name, module in auto_encoder.named_modules():
    if isinstance(module, torch.nn.Linear):
        module.register_forward_hook(save_activation(name))


# %%
#torch.seed()
torch.manual_seed(123465)
for n in range(10000):
    optimizer.zero_grad(set_to_none=True)
    batch_x, batch_y = get_batch_random(train_images_x, train_images_y, 80)
    batch_x = batch_x.detach().clone().requires_grad_(True)
    x, loss = auto_encoder(batch_x, batch_y, 'none')
    x.retain_grad()
    y_out = get_img_from_flatten(x)
    y_pred = get_img_from_flatten(batch_y)
    loss.backward()
    optimizer.step()
    if n % 5000 == 0:
        mask = (x > 0.99) | (x < 0.01)
        grads = x.grad * mask
        grads_img = grads.view(100, -1).cpu()

    if n % 50 == 0:
        with torch.no_grad():
            batch_x, batch_y = get_batch_random(test_images_x, test_images_y, 80)
            x, _ = auto_encoder(batch_x, batch_y, 'none')
        print(f'epoch {n}: train loss: {loss}, validation: {_}')
    
#%%
dead_neurons = {}
list_act = []
list_act_flatten = []
for name, act in activations.items():
    list_act.append(act.view(-1).tolist())

for act_list in list_act:
    for act in act_list:
        list_act_flatten.append(act)

acts_tensor = torch.tensor(list_act_flatten).unsqueeze(dim=0)
acts_tensor = acts_tensor.view(-1, 100)
mask = (acts_tensor > 0.99) | (acts_tensor < 0.01)
plt.imshow(mask, cmap='gray')

#%%
with torch.no_grad():
    torch.manual_seed(123465)
    data_point_x, data_point_y = get_batches(test_images_x, test_images_y, 10)
    y, _ = auto_encoder(data_point_x, None, 'none')
    y = y.cpu()
    x_in = get_img_from_flatten(data_point_x.cpu())
    y_in = get_img_from_flatten(data_point_y.cpu())
    y = get_img_from_flatten(y)
#    plt.imshow(x_in[0], cmap='gray')
#    plt.imshow(y_in[0], cmap='gray')
    plt.imshow(y[0], cmap='gray')

# %%
