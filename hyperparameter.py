import torch
#batch_size
batch_size = 32
#learning rate
lr=0.001
#device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#epochs for training
epochs=20



