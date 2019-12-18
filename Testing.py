import numpy as np
import functools
from PIL import Image
import IPython.display
import time
import cv2

import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import utils

# check that a gpu is available
torch.cuda.device_count()

parser = utils.prepare_parser()
parser = utils.add_sample_parser(parser)
config = vars(parser.parse_args('')) # use default arguments

config["resolution"] = utils.imsize_dict["I128_hdf5"]
config["n_classes"] = utils.nclass_dict["I128_hdf5"]
config["G_activation"] = utils.activation_dict["inplace_relu"]
config["D_activation"] = utils.activation_dict["inplace_relu"]
config["G_attn"] = "64"
config["D_attn"] = "64"
config["G_ch"] = 96
config["D_ch"] = 96
config["hier"] = True
config["dim_z"] = 120
config["shared_dim"] = 128
config["G_shared"] = True
config = utils.update_config_roots(config)
config["skip_init"] = True
config["no_optim"] = True
config["device"] = "cuda"
config['batch_size'] = 4

# Seed RNG.
utils.seed_rng(config["seed"])

# Set up cudnn.benchmark for free speed.
torch.backends.cudnn.benchmark = True

# Import the model.
model = __import__(config["model"])
experiment_name = utils.name_from_config(config)
G = model.Generator(**config).to(config["device"])
utils.count_parameters(G)

# Load weights.
G.load_state_dict(torch.load(weights_path))

# Update batch size setting used for G
# And prepare z and y distributions
G_batch_size = max(config["G_batch_size"], config["batch_size"])
(z_, y_) = utils.prepare_z_y(
    G_batch_size,
    G.dim_z,
    config["n_classes"],
    device=config["device"],
    fp16=config["G_fp16"],
    z_var=config["z_var"],
)

G.eval();

# Generate Random Samples

# We don't need gradients for this step since we are not training anything,
# so we wrap the forward pass with torch.no_grad
with torch.no_grad():
    z_.sample_()
    y_.sample_()
    image_tensors = G(z_, G.shared(y_))
    image_grid = torchvision.utils.make_grid(
        image_tensors,
        nrow=int(G_batch_size),
        normalize=True,
    )
    
    image_grid_np = image_grid.cpu().numpy().transpose(1, 2, 0) * 255
image_grid_np = np.uint8(image_grid_np)
print("Image Grid Shape: {}".format(np.shape(image_grid_np)))
print("Max pixel value: {}".format(np.max(image_grid_np)))
print("Min pixel value: {}".format(np.min(image_grid_np)))

IPython.display.display(Image.fromarray(image_grid_np))
