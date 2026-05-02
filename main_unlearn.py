import torch
from src.dataset import get_mnist
from src.unlearn import unlearn

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = get_mnist()

# choose any index to remove
data_index = 12345

unlearn(data_index, dataset, device)