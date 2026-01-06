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
target_img_size = [28, 28]
train_size = 0.9
channels = 1
flatten_size = target_img_size[0] *  target_img_size[1] * channels
latent_space_dim = 8 
latent_space_size =16 

#torch.manual_seed(329846)
device =  'cuda' if torch.cuda.is_available() else 'cpu'

# %%
images_path = './images_data/mnist/trainingSet/0'
paths = []
all_images = []

all_paths = Path(images_path)
#for directory in all_paths.iterdir():
#    paths.append(Path(directory))

paths.append(Path('./images_data/mnist/trainingSet/0'))
paths.append(Path('./images_data/mnist/trainingSet/1'))
paths.append(Path('./images_data/mnist/trainingSet/2'))
paths.append(Path('./images_data/mnist/trainingSet/3'))
paths.append(Path('./images_data/mnist/trainingSet/4'))

for p in paths:
    for item in p.iterdir():
        all_images.append(TF.resize(read_image(item.resolve()), target_img_size)[:3, :, :])


img_tensor_uint8 = torch.stack(all_images).to(device=device)

rand_indices = torch.randint(0, img_tensor_uint8.shape[0], (img_tensor_uint8.shape[0], ))
img_tensor_uint8 = img_tensor_uint8[rand_indices, :, :, :]

print(img_tensor_uint8.shape)
                                

# %%
imgs_tensor = img_tensor_uint8.float().to(device=device) / 255.0

#H,W,C
imgs_tensor = imgs_tensor.transpose(dim0=1, dim1=-1).transpose(dim0=1, dim1=2).to(device=device)
if channels != 1:
    imgs_tensor = imgs_tensor[:, :, :, 3 - channels:]

print(imgs_tensor.shape)
train_size_int = int(imgs_tensor.shape[0] * train_size)

train_images_x = imgs_tensor[:train_size_int].clone().to(device=device)
test_images_x = imgs_tensor[train_size_int:].clone().to(device=device)

train_images_y = imgs_tensor[:train_size_int].to(device=device)
test_images_y = imgs_tensor[train_size_int:].to(device=device)


torch.manual_seed(123465)
mu = 0
std_train = 0.5




#we assume that 0 is the batch dim
def get_batch_random(src_x, src_y, batch_size, flatten = True, channels_first_dim = True):
    batches_to_pick = torch.randint(0, src_x.shape[0], (batch_size, )).to(device=device)
    batches_x = src_x[batches_to_pick].clone().to(device=device)
    batches_y = src_y[batches_to_pick].clone().to(device=device)
    batches_x = batches_x + torch.randn_like(batches_x) * std_train
    if flatten == True:
        batches_x = batches_x.reshape(batch_size, -1).to(device=device)
        batches_y = batches_y.reshape(batch_size, -1).to(device=device)
    if channels_first_dim == True:
        batches_x =torch.transpose(batches_x, dim0= -1, dim1=1) 
    return batches_x, batches_y

def get_batches(src_x, src_y, batch_size = -1, flatten = True, channels_first_dim = True):
    if batch_size == -1:
        #pick all batches
        batch_size = src_x.shape[0]
    batches_x = src_x[:batch_size].clone().to(device= device)
    batches_y = src_y[:batch_size].clone().to(device= device)
    batches_x = batches_x + torch.randn_like(batches_x) * std_train
    if flatten == True:
        batches_x = batches_x.reshape(batch_size, -1).to(device= device)
        batches_y = batches_y.reshape(batch_size, -1).to(device= device)
    if channels_first_dim == True:
        batches_x =torch.transpose(batches_x, dim0= -1, dim1=1) 
    return batches_x, batches_y

def get_batch_sliced(src_x, src_y, batch_size, batch_idx, flatten = True, channels_first_dim = True):
    if batch_size == -1:
        #pick all batches
        batch_size = src_x.shape[0]
    offset = batch_size * batch_idx 
    batches_x = src_x[offset:offset + batch_size].clone().to(device= device)
    batches_y = src_y[offset:offset + batch_size].clone().to(device= device)
    batches_x = batches_x + torch.randn_like(batches_x) * std_train
    if flatten == True:
        batches_x = batches_x.reshape(batch_size, -1).to(device= device)
        batches_y = batches_y.reshape(batch_size, -1).to(device= device)
    if channels_first_dim == True:
        batches_x =torch.transpose(batches_x, dim0= -1, dim1=1) 
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
            torch.nn.Linear(flatten_size, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(0.2),
            torch.nn.Sigmoid(),
            torch.nn.Linear(128, latent_space_size),
            torch.nn.Sigmoid(),
            torch.nn.Linear(latent_space_size, 1),
       )
    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(1, latent_space_size),
            torch.nn.Linear(latent_space_size, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(512, flatten_size),
            torch.nn.Dropout(0.2),
            torch.nn.Sigmoid(),
        )
    def forward(self, x):
        return self.model(x)

class AutoEncoder(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    def encode(self, x):
        return self.encoder(x)
    def decode(self, x):
        return self.decoder(x)

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
                loss = F.mse_loss(x, torch.flatten(target, 1))
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
batch_size = 1000
batch_count = int(float(train_images_x.shape[0]) / 1000.0)
batch_count = 1
print(batch_count)

# %%
torch.manual_seed(123465)
for n in range(1000):
    optimizer.zero_grad(set_to_none=True)
    batch_x, batch_y = get_batch_random(train_images_x, train_images_y, batch_size)
#        batch_x = batch_x.detach().clone().requires_grad_(True)
    x, loss = auto_encoder(batch_x, batch_y, 'none')
    x.retain_grad()
    loss.backward()
    optimizer.step()
    if n % 50 == 0:
        with torch.no_grad():
            auto_encoder.eval()
            batch_x, batch_y = get_batches(test_images_x, test_images_y)
            x, _ = auto_encoder(batch_x, batch_y, 'none')
            print(f'epoch {n}: train loss: {loss}, validation: {_}')
    auto_encoder.train(True)
    
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
acts_tensor = acts_tensor.view(-1, 1000)
mask = (acts_tensor > 0.99) | (acts_tensor < 0.01)
plt.imshow(mask, cmap='gray')

#%%
with torch.no_grad():
    auto_encoder.eval()
    torch.manual_seed(123465)
    data_point_x, data_point_y = get_batches(test_images_x, test_images_y, batch_size=-1, flatten=True)
    y, _ = auto_encoder(data_point_x, None, 'none')
    y = y.cpu()
    x_in = get_img_from_flatten(data_point_x.cpu())
    y_in = get_img_from_flatten(data_point_y.cpu())
    y = get_img_from_flatten(y)
#    plt.imshow(x_in[0], cmap='gray')
#    plt.imshow(y_in[0], cmap='gray')
    plt.imshow(y[0], cmap='gray')
    print(F.mse_loss(y, y_in))

# %%

with torch.no_grad():
    auto_encoder.eval()
    data_point_x, data_point_y = get_batches(test_images_x, test_images_y, batch_size=-1, flatten=True)
    latent_data = auto_encoder.encode(data_point_x)
    t = torch.linspace(0, 1, 100).to(device=device)
    t_ease = t
    interp_latent = torch.lerp(latent_data[0], latent_data[2], t_ease[20])
    interp_latent = interp_latent.unsqueeze(dim = 0)
    decoded_data = auto_encoder.decode(interp_latent)
    x_test = get_img_from_flatten(data_point_x.cpu())
    y_test = get_img_from_flatten(data_point_y.cpu())
    y = get_img_from_flatten(decoded_data.cpu())
    plt.imshow(y[0], 'gray')


# %%
