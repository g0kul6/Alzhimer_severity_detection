#import necessary libs
import os
import glob
import random
import numpy as np
from torch.utils.data import random_split
import torch


#test paths
test_Mild='Alzheimers Dataset/test/MildDemented/'
test_Moderate='Alzheimers Dataset/test/ModerateDemented/'
test_Non='Alzheimers Dataset/test/NonDemented/'
test_VeryMild='Alzheimers Dataset/test/VeryMildDemented/'

#train_paths
train_Mild='Alzheimers Dataset/train/MildDemented/'
train_Moderate='Alzheimers Dataset/train/ModerateDemented/'
train_Non='Alzheimers Dataset/train/NonDemented/'
train_VeryMild='Alzheimers Dataset/train/VeryMildDemented/'

#list to train imgs path
train_Mild_list = glob.glob(os.path.join(train_Mild,'*.jpg'))
train_Moderate_list= glob.glob(os.path.join(train_Moderate,'*.jpg'))
train_Non_list= glob.glob(os.path.join(train_Non,'*.jpg'))
train_VeryMild_list= glob.glob(os.path.join(train_VeryMild,'*.jpg'))

#list to test imgs path
test_Mild_list = glob.glob(os.path.join(test_Mild,'*.jpg'))
test_Moderate_list= glob.glob(os.path.join(test_Moderate,'*.jpg'))
test_Non_list= glob.glob(os.path.join(test_Non,'*.jpg'))
test_VeryMild_list= glob.glob(os.path.join(test_VeryMild,'*.jpg'))

#list of train and test img paths
train_list=train_Mild_list+train_Moderate_list+train_Non_list+train_VeryMild_list
test_list=test_Mild_list+test_Moderate_list+test_Non_list+test_VeryMild_list

#random shuffle of train_list
random.shuffle(train_list)

#val train split(0.9 and 0.1)
l=len(train_list)
val=0.1
val_len=int(val*l)
train_len=l-val_len
train_list,val_list=random_split(train_list,[train_len,val_len],generator=torch.Generator().manual_seed(42))

#saving the train,val and test list of paths
np.save('Data_List/train_list',train_list)
np.save('Data_List/val_list',val_list)
np.save('Data_List/test_list',test_list)


