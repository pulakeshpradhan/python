---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: torch
    language: python
    name: python3
---

## Import libraries

```python
import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## ConvLSTM

```python
class ConvLSTMCell(nn.Module):
    
    def __init__(self,input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        
    def __initStates(self, size, device):
        return torch.zeros(size).to(device), torch.zeros(size).to(device)
    
    def forward(self, input_tensor, cur_state):
        if cur_state == None:
            h_cur, c_cur = self.__initStates([input_tensor.shape[0], self.hidden_dim, input_tensor.shape[2], input_tensor.shape[3]], device=input_tensor.device)
        else:
            h_cur, c_cur = cur_state
            
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))
```

```python
# EXAMPLE USAGE
# Parameters
batch_size = 2
input_dim = 3
hidden_dim = 16
height, width = 32, 32
kernel_size = (3, 3)

# Create dummy input (one time step)
x = torch.randn(batch_size, input_dim, height, width).to(device)

# Initialize ConvLSTMCell
cell = ConvLSTMCell(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    kernel_size=kernel_size,
    bias=True,
).to(device)

# Forward pass (first time step)
h_next, c_next = cell(x, cur_state=None)

print("h_next shape:", h_next.shape)
print("c_next shape:", c_next.shape)

# Second time step
x2 = torch.randn(batch_size, input_dim, height, width).to(device)

h2, c2 = cell(x2, cur_state=(h_next, c_next))

print("h2 shape:", h2.shape)
print("c2 shape:", c2.shape)
```

```python
class ConvLSTM(nn.Module):
    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super().__init__()
        
        self._check_kernel_size_consistency(kernel_size)
        
        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError("Inconsistent list length.")
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))
        
        self.cell_list = nn.ModuleList(cell_list)
        
    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
            
        b, _, _, h, w = input_tensor.size()
        
        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))
            
        layer_output_list = []
        last_state_list = []
        
        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor
        
        for layer_idx in range(self.num_layers):
            
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)
                
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])
            
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            layer_state_list = last_state_list[-1:]
            
        return layer_output_list, layer_state_list
                
    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states
            
    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError("'kernel_size' must be tuple or list of tuples.")
        
    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
```

```python
# EXAMPLE USAGE
# Parameters
batch_size = 2
time_steps = 5
input_dim = 3
hidden_dim = 16
height, width = 32, 32
kernel_size = (3, 3)
num_layers = 1

# Dummy input (B, T, C, H, W)
x = torch.randn(batch_size, time_steps, input_dim, height, width).to(device)

# Model
model = ConvLSTM(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    kernel_size=kernel_size,
    num_layers=num_layers,
    batch_first=True,
    bias=True,
    return_all_layers=False
).to(device)

# Forward pass
layer_outputs, last_states = model(x)

print("Layer outputs length:", len(layer_outputs))
print("Last states length:", len(last_states))
print("Output shape:", layer_outputs[0].shape)
print("Hidden state shape:", last_states[0][0].shape)
print("Cell state shape:", last_states[0][1].shape)
```

## Generator

```python
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
```

```python
# EXAMPLE USAGE
states_encoder = [None] * 7
states_decoder = [None] * 7
x = torch.randn((2, 4, 128, 128, 10), dtype=torch.float32) 
model = Generator(in_channels=4, out_channels=4)
y, states = model(x)
print(y.size())
```

## Discriminator

```python
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
```

```python
# EXAMPLE USAGE
x = torch.randn((16, 3, 32, 32, 4), dtype=torch.float32) 
model = Discriminator(in_channels=3)
y = model(x)
print(y.size())
```
