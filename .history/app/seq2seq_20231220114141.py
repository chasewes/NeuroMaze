import torch.nn as nn
import torch

device = 'cpu'

class ConvLSTMCell(nn.Module):

    def __init__(self, in_channels, out_channels, 
    kernel_size, padding, activation, frame_size):

        super(ConvLSTMCell, self).__init__()  

        if activation == "tanh":
            self.activation = torch.tanh 
        elif activation == "relu":
            self.activation = torch.relu
        
        total_in_channels = in_channels + out_channels
        self.conv = nn.Conv2d(in_channels=total_in_channels, out_channels=4 * out_channels, kernel_size=kernel_size, padding=padding)

        
        self.W_ci = nn.Parameter(torch.Tensor(out_channels, *frame_size).to(device))
        self.W_co = nn.Parameter(torch.Tensor(out_channels, *frame_size).to(device))
        self.W_cf = nn.Parameter(torch.Tensor(out_channels, *frame_size).to(device))

    def forward(self, X, H_prev, C_prev):

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        conv_output = self.conv(torch.cat([X, H_prev], dim=1))

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        i_conv, f_conv, C_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)

        input_gate = torch.sigmoid(i_conv + self.W_ci * C_prev )
        forget_gate = torch.sigmoid(f_conv + self.W_cf * C_prev )

        # Current Cell output
        C = forget_gate*C_prev + input_gate * self.activation(C_conv)

        output_gate = torch.sigmoid(o_conv + self.W_co * C )

        # Current Hidden State
        H = output_gate * self.activation(C)

        return H, C
    
    def reset_parameters(self):
        # Kaiming He initialization for ReLU
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv.bias, 0)

        # Xavier Glorot initialization for Tanh
        nn.init.xavier_normal_(self.W_ci)
        nn.init.xavier_normal_(self.W_co)
        nn.init.xavier_normal_(self.W_cf)

class ConvLSTM(nn.Module):
    
    def __init__(self, in_channels, out_channels, 
    kernel_size, padding, activation, frame_size):

        super(ConvLSTM, self).__init__()

        self.out_channels = out_channels

        # We will unroll this over time steps
        self.convLSTMcell = ConvLSTMCell(in_channels, out_channels, kernel_size, padding, activation, frame_size)


    def forward(self, X):
        batch_size, _, height, width = X.size()

        # Initialize output
        output = torch.zeros(batch_size, self.out_channels, height, width, device=device)
        
        # Initialize Hidden State
        H = torch.zeros(batch_size, self.out_channels, height, width, device=device)

        # Initialize Cell Input
        C = torch.zeros(batch_size, self.out_channels, height, width, device=device)

        # Process the single frame
        H, C = self.convLSTMcell(X, H, C)
        output = H

        return output
    
    def reset_parameters(self):
        self.convLSTMcell.reset_parameters()


class Seq2Seq(nn.Module):
    def __init__(self, num_channels, num_actions, num_kernels, kernel_size, padding, 
                 activation, frame_size, num_layers):

        super(Seq2Seq, self).__init__()

        in_channels = num_channels + num_actions  # RGB channels + action encodings

        self.sequential = nn.Sequential()

        # First layer with in_channels as 8
        self.sequential.add_module("convlstm1", ConvLSTM(in_channels=num_channels+num_actions, out_channels=num_kernels, kernel_size=kernel_size, padding=padding, activation=activation, frame_size=frame_size))

        for l in range(2, num_layers + 1):
            self.sequential.add_module(f"convlstm{l}", ConvLSTM(in_channels=num_kernels, out_channels=num_kernels, kernel_size=kernel_size, padding=padding, activation=activation, frame_size=frame_size))
                
            self.sequential.add_module(f"batchnorm{l}", nn.BatchNorm2d(num_features=num_kernels))


        # Add Convolutional Layer to predict output frame
        self.conv = nn.Conv2d(
            in_channels=num_kernels, out_channels=num_channels,  # Outputting RGB channels
            kernel_size=kernel_size, padding=padding)

    def forward(self, X):
        # Forward propagation through all the layers
        output = self.sequential(X)

        # Process the last output frame
        output = self.conv(output)
        
        return nn.Sigmoid()(output)
    
    def reset_parameters(self):
        for layer in self.sequential:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        # Initialize the final convolution layer
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv.bias, 0)