from image_preprocessing import train_loader,val_loader,test_loader
from resnet import model_18,optimizer_18,lo
from hyperparameter import device,epochs
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch
loss1=[]
loss2=[]
acc1=[]
acc2=[]
for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    
    for data, label in train_loader:
        data = data.cuda()
        
        label = label.cuda()
        
        output = model_18(data)
        loss = lo(output, label)
       
        train_list = F.softmax(output,dim=1).tolist()
        
        optimizer_18.zero_grad()
        loss.backward()
        optimizer_18.step()
        
        acc = ((output.argmax(dim=1) == label).float().mean())
        epoch_accuracy += acc/len(train_loader)
        epoch_loss += loss/len(train_loader)
    
    loss1.append(epoch_loss.item())
    acc1.append(epoch_accuracy.item())
    print('Epoch : {}, train accuracy : {}, train loss : {}'.format(epoch+1, epoch_accuracy,epoch_loss))
    with torch.no_grad():
        epoch_val_accuracy=0
        epoch_val_loss =0
        for data, label in val_loader:
            data = data.to(device)
            
            label = label.to(device)
            
            val_output = model_18(data)
            val_loss = lo(val_output,label)
            
            
            acc = ((val_output.argmax(dim=1) == label).float().mean())
            epoch_val_accuracy += acc/ len(val_loader)
            epoch_val_loss += val_loss/ len(val_loader)
        acc2.append(epoch_val_accuracy.item())
        loss2.append(epoch_val_loss.item())    
        print('Epoch : {}, val_accuracy : {}, val_loss : {}'.format(epoch+1, epoch_val_accuracy,epoch_val_loss))

torch.save(model_18.state_dict(),"trained_model_pth/iter_100.pth")