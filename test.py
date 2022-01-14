import torch
from resnet import model_18,lo
from image_preprocessing import val_loader,test_loader
import torch 
from hyperparameter import epochs,device
import cv2
import numpy as np
model=model_18.cuda()
model.load_state_dict(torch.load("trained_model_pth/iter_100.pth"))
model.eval()

with torch.no_grad():
        epoch_val_accuracy=0
        epoch_val_loss =0
        for data, label in test_loader:
            data = data.to(device)
            
            label = label.to(device)
            
            val_output = model(data)
            val_loss = lo(val_output,label)
            
            
            acc = ((val_output.argmax(dim=1) == label).float().mean())
            epoch_val_accuracy += acc/ len(test_loader)
            epoch_val_loss += val_loss/ len(test_loader)
