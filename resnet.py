import torch
import torch.nn as nn
import torch.optim as optim
from hyperparameter import lr,device

#block which is repeated in resnet-18 and resnet-34
class block_18_34(nn.Module):
    def __init__(self,in_channels,out_channels,identity_downsample=None,stride=1):
        super(block_18_34,self).__init__()
        self.expansion=4
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=0,bias=False)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU()
        self.identity_downsample=identity_downsample
        self.stride=stride
    def forward(self,x):
        identity=x.clone()

        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.conv2(x)
        x=self.bn2(x)
        
        if self.identity_downsample is not None:
            identity=self.identity_downsample(identity)
        
        x+=identity
        x=self.relu(x)
        return x

#resnet class
class resnet(nn.Module):
    def __init__(self,block,layers,img_channels,num_classes):
        super().__init__()
        self.in_channels=64
        self.conv1=nn.Conv2d(img_channels,64,kernel_size=7,stride=1,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU()
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        #resnet layers
        self.layer_1=self.layer_n(block,layers[0],64,stride=1)
        self.layer_2=self.layer_n(block,layers[1],128,stride=2)
        self.layer_3=self.layer_n(block,layers[2],256,stride=2)
        self.layer_4=self.layer_n(block,layers[3],512,stride=2)
        

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512,num_classes)
    
    def layer_n(self,block,num_residual_blocks,out_channels,stride):
        layers=[]
        identity_downsample=nn.Sequential(nn.Conv2d(self.in_channels,out_channels,kernel_size=1,stride=stride),
                                          nn.BatchNorm2d(out_channels))
        layers.append(block(self.in_channels,out_channels,identity_downsample,stride))
        
        #rest of residual layers not required with downsample already mapped to required dimension
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self,x):
        #starting 7x7,maxpool layers
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)

        #residual blocks
        x=self.layer_1(x)
        x=self.layer_2(x)
        x=self.layer_3(x)
        x=self.layer_4(x)

        #avgpool and fully_connected
        x=self.avgpool(x)
        x=self.fc(x)

        return x

#resnet-18 and resnet-34
model_18=resnet(block_18_34,[2,2,2,2],1,4).to(device)
#model_34=resnet(block_18_34,[3,4,6,3],1,4).to(device)
model_18.train()
#model_34.train()
#adam optimizers
optimizer_18=optim.Adam(params=model_18.parameters(),lr=lr)
#optimizer_34=optim.Adam(params=model_34.parameters(),lr=lr)
#cross entropy loss
loss=nn.CrossEntropyLoss()
