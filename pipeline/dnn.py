import torch
import torch.nn as nn
import numpy as np

class FCNN(nn.Module):
    def __init__(self, num_hidden=1, dropout_rate=0.3, input_length = 50, num_input_channels = 63):

        super().__init__()

        self.input_length = input_length
        self.num_input_channels = num_input_channels

        self.num_hidden = num_hidden
        units = np.round(np.linspace(1, self.input_length*self.num_input_channels, self.num_hidden+2)[::-1]).astype(int)
        self.fully_connected = torch.nn.ModuleList([torch.nn.Linear(units[i], units[i+1]) for i in range(len(units)-1)])
        self.activations = torch.nn.ModuleList([torch.nn.Tanh() for i in range(len(units)-2)])
        self.dropouts = torch.nn.ModuleList([torch.nn.Dropout(p=dropout_rate) for i in range(len(units)-2)])

    def __str__(self):
        return 'fcnn'

    def forward(self, x):
        x = x.flatten(start_dim=1)
        for i, layer in enumerate(self.fully_connected[:-1]):
          x = layer(x)
          x = self.activations[i](x)
          x = self.dropouts[i](x)
        
        x = self.fully_connected[-1](x)
        return x.flatten()

class CNN(nn.Module):

    def __init__(self,
                 F1=16,
                 D=16,
                 F2=16,
                 dropout_rate=0.25,
                 input_length = 50,
                 num_input_channels = 63):

        super().__init__()

        self.input_length = input_length
        self.num_input_channels = num_input_channels
 
        self.F1=F1
        self.F2=F2
        self.D=D

        self.conv1_kernel=3
        self.conv3_kernel=3
        self.temporalPool1 = 2
        self.temporalPool2 = 5

        # input shape is [1, C, T]
        
        self.conv1 = torch.nn.Conv2d(1, self.F1, (1, self.conv1_kernel), padding='same')
        self.conv2 = torch.nn.Conv2d(self.F1, self.F1*self.D, (self.num_input_channels, 1), padding='valid', groups=self.F1)
        self.conv3 = torch.nn.Conv2d(self.F1*self.D, self.F1*self.D, (1, self.conv1_kernel), padding='same', groups=self.F1*self.D)
        self.conv4 = torch.nn.Conv2d(self.F1*self.D, self.F2, (1,1))

        self.pool1 = torch.nn.AvgPool2d((1, self.temporalPool1))
        self.pool2 = torch.nn.AvgPool2d((1, self.temporalPool2))
                
        self.linear = torch.nn.Linear(self.F2*self.input_length//(self.temporalPool1*self.temporalPool2), 1)

        self.bnorm1 = torch.nn.BatchNorm2d(self.F1)
        self.bnorm2 = torch.nn.BatchNorm2d(self.F1*self.D)
        self.bnorm3 = torch.nn.BatchNorm2d(F2)

        self.dropout1 = torch.nn.Dropout2d(dropout_rate)
        self.dropout2 = torch.nn.Dropout2d(dropout_rate)

        self.activation1 = torch.nn.ELU()
        self.activation2 = torch.nn.ELU()
        self.activation3 = torch.nn.ELU()

    def __str__(self):
        return 'cnn'
     
    def forward(self, x):
        #x shape = [batch, C, T]
        x = x.unsqueeze(1)
        out = self.conv1(x)
        out = self.bnorm1(out)

        out = self.conv2(out)
        out = self.bnorm2(out)
        out = self.activation1(out)
        out = self.pool1(out)
        out = self.dropout1(out)

        #shape is now [batch, DxF1, 1, T//TPool1]
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.bnorm3(out)
        out = self.activation2(out)
        out = self.pool2(out)
        out = self.dropout2(out)
        
        out = torch.flatten(out, start_dim = 1) # shape is now [batch, F2*T//(TPool1*TPool2)]
        out = self.linear(out)
        return out.flatten()