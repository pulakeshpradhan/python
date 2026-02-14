import sys
import torch
from torch import nn
import torch.nn.functional as F

sys.path.append('src/models')
from convlstm import ConvLSTMCell

class Generator(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1)    
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(hidden_dim * 4, hidden_dim * 8, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(hidden_dim * 8, hidden_dim * 8, kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.Conv2d(hidden_dim * 8, hidden_dim * 8, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(hidden_dim * 8, hidden_dim * 8, kernel_size=3, stride=2, padding=1)

        self.conv_lstm_e1 = ConvLSTMCell(hidden_dim, hidden_dim, kernel_size=(3, 3), bias=False)
        self.conv_lstm_e2 = ConvLSTMCell(hidden_dim * 2, hidden_dim * 2, kernel_size=(3, 3), bias=False)
        self.conv_lstm_e3 = ConvLSTMCell(hidden_dim * 4, hidden_dim * 4, kernel_size=(3, 3), bias=False)
        self.conv_lstm_e4 = ConvLSTMCell(hidden_dim * 8, hidden_dim * 8, kernel_size=(3, 3), bias=False)
        self.conv_lstm_e5 = ConvLSTMCell(hidden_dim * 8, hidden_dim * 8, kernel_size=(3, 3), bias=False)
        self.conv_lstm_e6 = ConvLSTMCell(hidden_dim * 8, hidden_dim * 8, kernel_size=(3, 3), bias=False)
        self.conv_lstm_e7 = ConvLSTMCell(hidden_dim * 8, hidden_dim * 8, kernel_size=(3, 3), bias=False)    

        self.conv_lstm_d1 = ConvLSTMCell(hidden_dim * 8, hidden_dim * 8, kernel_size=(3, 3), bias=False)
        self.conv_lstm_d2 = ConvLSTMCell(hidden_dim * 8 * 2, hidden_dim * 8, kernel_size=(3, 3), bias=False)
        self.conv_lstm_d3 = ConvLSTMCell(hidden_dim * 8 * 2, hidden_dim * 8, kernel_size=(3, 3), bias=False)
        self.conv_lstm_d4 = ConvLSTMCell(hidden_dim * 8 * 2, hidden_dim * 4, kernel_size=(3, 3), bias=False)
        self.conv_lstm_d5 = ConvLSTMCell(hidden_dim * 4 * 2, hidden_dim * 2, kernel_size=(3, 3), bias=False)
        self.conv_lstm_d6 = ConvLSTMCell(hidden_dim * 2 * 2, hidden_dim, kernel_size=(3, 3), bias=False)
        self.conv_lstm_d7 = ConvLSTMCell(hidden_dim * 2, hidden_dim, kernel_size=(3, 3), bias=False)
        
        self.up = nn.Upsample(scale_factor=2)
        self.conv_out = nn.Conv2d(hidden_dim, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.slope = 0.2
        
    def weight_init(self, mean, std):
        for m in self.modules():
            normal_init(m, mean, std)
            
    def forward_step(self, input, states_encoder, states_decoder):
        e1 = self.conv1(input)
        states_e1 = self.conv_lstm_e1(e1, states_encoder[0])
        e2 = self.conv2(F.leaky_relu(states_e1[0], self.slope))
        states_e2 = self.conv_lstm_e2(e2, states_encoder[1])
        e3 = self.conv3(F.leaky_relu(states_e2[0], self.slope))
        states_e3 = self.conv_lstm_e3(e3, states_encoder[2])
        e4 = self.conv4(F.leaky_relu(states_e3[0], self.slope))
        states_e4 = self.conv_lstm_e4(e4, states_encoder[3])
        e5 = self.conv5(F.leaky_relu(states_e4[0], self.slope))
        states_e5 = self.conv_lstm_e5(e5, states_encoder[4])
        e6 = self.conv6(F.leaky_relu(states_e5[0], self.slope))
        states_e6 = self.conv_lstm_e6(e6, states_encoder[5])
        e7 = self.conv7(F.leaky_relu(states_e6[0], self.slope))
        states_e7 = self.conv_lstm_e7(e7, states_encoder[6])
        
        states_d1 = self.conv_lstm_d1(F.relu(states_e7[0]), states_decoder[0])
        d1 = self.up(states_d1[0])
        d1 = torch.cat([d1, e6], 1)
        
        states_d2 = self.conv_lstm_d2(F.relu(d1), states_decoder[1])
        d2 = self.up(states_d2[0])
        d2 = torch.cat([d2, e5], 1)
        
        states_d3 = self.conv_lstm_d3(F.relu(d2), states_decoder[2])
        d3 = self.up(states_d3[0])
        d3 = torch.cat([d3, e4], 1) 
        
        states_d4 = self.conv_lstm_d4(F.relu(d3), states_decoder[3])
        d4 = self.up(states_d4[0])
        d4 = torch.cat([d4, e3], 1) 
        
        states_d5 = self.conv_lstm_d5(F.relu(d4), states_decoder[4])
        d5 = self.up(states_d5[0])
        d5 = torch.cat([d5, e2], 1)
        
        states_d6 = self.conv_lstm_d6(F.relu(d5), states_decoder[5])
        d6 = self.up(states_d6[0])
        d6 = torch.cat([d6, e1], 1) 
        
        states_d7 = self.conv_lstm_d7(F.relu(d6), states_decoder[6])
        d7 = self.up(states_d7[0])
        
        out = torch.tanh(self.conv_out(d7))
        
        states_e = [states_e1, states_e2, states_e3, states_e4, states_e5, states_e6, states_e7]
        states_d = [states_d1, states_d2, states_d3, states_d4, states_d5, states_d6, states_d7]
        
        return out, (states_e, states_d)
        
    def forward(self, x):
        states_encoder = [None] * 7
        states_decoder = [None] * 7
        batch, _, h, w, t = x.shape
        output = torch.zeros((batch, self.conv_out.out_channels, h, w, t), device=x.device)

        for t in range(x.shape[4]):
            output[..., t], states = self.forward_step(x[..., t], states_encoder, states_decoder)
            states_encoder, states_decoder = states[0], states[1]
        return output, states
    
def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

if __name__== '__main__':
    states_encoder = [None] * 7
    states_decoder = [None] * 7
    x = torch.randn((2, 4, 128, 128, 10), dtype=torch.float32) 
    model = Generator(in_channels=4, out_channels=4)
    y, states = model(x)
    print(y.size())