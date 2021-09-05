import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
from PIL import Image,ImageOps
from hyperparameter import batch_size
import cv2

#tranformations to train images
train_transforms=transforms.Compose([transforms.Resize((224,224)),
                                    
                                    transforms.ToTensor()])
#transformation to val images
val_transforms=transforms.Compose([transforms.Resize((224, 224)),
                                  
                                  transforms.ToTensor()])
#transformation to test imagese
test_transforms=transforms.Compose([transforms.Resize((224, 224)),
                                   
                                   transforms.ToTensor()])   

class dataset(Dataset):
  def __init__(self,file_list,transform=None):
    self.file_list=file_list
    self.transform=transform
  def __len__(self):
    self.filelength=len(self.file_list)
    return self.filelength
  def __getitem__(self,idx):
    img_path=self.file_list[idx]
    img=Image.open(img_path)
    img=ImageOps.grayscale(img)
    img_transformed=self.transform(img)
    #get the lable 
    lable=img_path.split('\\')[2]
    #one-hot encoding the lable
    if lable=='MildDemented':
        lable=0
    elif lable=='ModerateDemented':
        lable=1
    elif lable=='NonDemented':
        lable=2
    elif lable=='VeryMildDemented':
        lable=3
    return img_transformed,lable

#loading the train,test and val lists
train_list=np.load('train_list.npy')
test_list=np.load('test_list.npy')
val_list=np.load('val_list.npy')


#creating the data with lables
train_data=dataset(train_list,transform=train_transforms)
val_data=dataset(val_list,transform=val_transforms)
test_data=dataset(test_list,transform=test_transforms)


#creating data loader for train val and test
train_loader=DataLoader(dataset = train_data,batch_size=batch_size,shuffle=True)
val_loader=DataLoader(dataset = val_data,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(dataset = test_data,batch_size=batch_size,shuffle=True)


"""
Lables:
    MildDemented->0
    ModerateDemented->1
    NonDemented->2
    VeryMildDemented->3
"""
