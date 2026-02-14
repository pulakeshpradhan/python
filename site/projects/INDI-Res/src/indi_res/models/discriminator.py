import torch
from torch import nn
import torch.nn.functional as F

from convlstm import ConvLSTMCell

class Discriminator(nn.Module):
    def __init__(self, in_channels=4, hidden_dim=64):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1)
        self.conv2_bn = nn.BatchNorm2d(hidden_dim * 2, track_running_stats=False)
        self.conv3 = nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1)
        self.conv3_bn = nn.BatchNorm2d(hidden_dim * 4, track_running_stats=False)
        self.conv4_lstm = ConvLSTMCell(hidden_dim * 4, hidden_dim * 4, kernel_size=(3, 3), bias=False)
        self.conv4 = nn.Conv2d(hidden_dim * 4, hidden_dim * 8, kernel_size=4, stride=1, padding=1)
        self.conv4_bn = nn.BatchNorm2d(hidden_dim * 8, track_running_stats=False)
        self.conv5_lstm = ConvLSTMCell(hidden_dim * 8, hidden_dim * 8, kernel_size=(3, 3), bias=False)
        self.conv5 = nn.Conv2d(hidden_dim * 8, 1, kernel_size=4, stride=1, padding=1)
                
        self.slope = 0.2
        
        torch.backends.cudnn.deterministic = True
        
    def weight_init(self, mean, std):
        for m in self.modules():
            normal_init(m, mean, std)
            
    def forward_step(self, input, states):
        x = F.leaky_relu(self.conv1(input), self.slope)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), self.slope)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), self.slope)
        states1 = self.conv4_lstm(x, states[0])
        x = F.leaky_relu(self.conv4_bn(self.conv4(states1[0])), self.slope)
        states2 = self.conv5_lstm(x, states[1])
        x = F.leaky_relu(self.conv5(states2[0]), self.slope)
        
        return x.squeeze(dim=1), (states1, states2) 
        
    def forward(self, x):
        batch, _, h, w, t = x.shape
        output = torch.zeros((batch, (h//8)-2, (w//8)-2, t), device=x.device)
        states = (None, None)
        for timestep in range(t):
            output[..., timestep], states = self.forward_step(x[..., timestep], states)
        return F.sigmoid(output)
    
def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
        
if __name__ == '__main__':
    x = torch.randn((16, 3, 32, 32, 4), dtype=torch.float32) 
    model = Discriminator(in_channels=3)
    y = model(x)
    print(y.size())