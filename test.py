from email.mime import image
import torch
from resnet import model_18,lo
from image_preprocessing import val_loader,test_loader,train_loader
import torch 
from hyperparameter import epochs,device
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

model=model_18.cuda()
model.load_state_dict(torch.load("trained_model_pth/iter_100.pth"))
model.eval()
labels=[]
outputs=[]
with torch.no_grad():
        epoch_val_accuracy=0
        epoch_val_loss =0
        for data, label in val_loader:
            data = data.to(device)
            
            label = label.to(device)
            
            output = model(data)
            loss = lo(output,label)
            for i in range(len(label)):
                labels.append(label[i].item())
                outputs.append(torch.argmax(output[i]).item())
            
            acc = ((output.argmax(dim=1) == label).float().mean())
            epoch_val_accuracy += acc/ len(test_loader)
            epoch_val_loss += loss/ len(test_loader)
            

conf_matrix=confusion_matrix(labels,outputs)
fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()
