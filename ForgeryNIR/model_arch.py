import torchvision.models as models
import torch.nn as nn
import torch.cuda
from torch.nn import TransformerEncoderLayer


def ResNet_50():
    m=models.resnet50()
    new_model = nn.Sequential(*list(m.children())[:-1])
    resnet_50 = new_model
    return resnet_50

def ResNet_50_dct():
    resnet_50 = models.resnet50()
    resnet_50.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    new_model = nn.Sequential(*list(resnet_50.children())[:-1])
    resnet_50 = new_model
    return resnet_50

class AttentionResNet(nn.Module):
    def __init__(self,n_classes):
        super(AttentionResNet, self).__init__()
        self.fc1=nn.Linear(18432,2048)
        self.dropout=nn.Dropout(0.5)
        self.fc2=nn.Linear(2048,512)
        self.fc3 = nn.Linear(512, n_classes)
        self.act=nn.ReLU()
        self.attention = TransformerEncoderLayer(d_model=2048,nhead=8)
    def forward(self,x):
        x=x.view(-1,9,2048)
        x=self.attention(x)
        x=x.view(-1,18432)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class R_MLP(nn.Module):
    def __init__(self,n_classes):
        super(R_MLP, self).__init__()
        self.fc1=nn.Linear(2048,512)
        self.dropout=nn.Dropout(0.5)
        self.fc2=nn.Linear(512,512)
        self.fc3 = nn.Linear(512, n_classes)
        self.act=nn.ReLU()

    def forward(self,x):
        x=x.view(-1,2048)
        x=self.fc1(x)
        x=self.act(x)
        x=self.dropout(x)
        x=self.fc2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class SIFTMLP(nn.Module):
    def __init__(self,n_classes):
        super(SIFTMLP, self).__init__()
        self.fc1=nn.Linear(8704,2048)
        self.dropout=nn.Dropout(0.5)
        self.fc2=nn.Linear(2048,512)
        self.fc3 = nn.Linear(512, n_classes)
        self.act=nn.Hardswish(inplace=True)
        self.transformer_encoder = TransformerEncoderLayer(d_model=128,nhead=8).cuda()
    def forward(self,x):
        x=x.view(-1,68,128)
        x=self.transformer_encoder(x)
        x=x.view(-1,8704)
        x=self.fc1(x)
        x=self.act(x)
        x=self.dropout(x)
        x=self.fc2(x)
        x=self.act(x)
        x = self.dropout(x)
        x=self.fc3(x)
        return x


