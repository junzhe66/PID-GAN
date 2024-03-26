import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.tokenizer import Tokenizer


class ReLUClamp(nn.Module):
    def __init__(self, max_value=1):
        super(ReLUClamp, self).__init__()
        self.max_value = max_value

    def forward(self, x):
        return F.relu(x).clamp(max=self.max_value)


def conv_block(in_channels, out_channels, kernel_size, stride, relu_alpha=0.2, 
               transposed=False, output_layer=False, batch_norm=False, drop_out=False):
    # Create convolutional layers
    layers = []

    padding = (kernel_size - 1) // 2
    conv_layer = nn.Conv2d if not transposed else nn.ConvTranspose2d
    layers.append(conv_layer(in_channels, out_channels, kernel_size, stride, padding=padding))

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    if output_layer:
        layers.append(ReLUClamp(max_value=1))
    else:
        layers.append(nn.LeakyReLU(relu_alpha, inplace=True))
        if drop_out:
            layers.append(nn.Dropout(0.2))

    return nn.Sequential(*layers)

class DiscriminatorAENN(nn.Module):
    def __init__(self, relu_alpha=0.2, batch_norm=False, drop_out=False):
        super().__init__()
        self.relu_alpha = relu_alpha
        self.batch_norm = batch_norm
        self.drop_out = drop_out

        self.conv_model = nn.Sequential(
            conv_block(1, 32, 5, stride=2, relu_alpha=relu_alpha, batch_norm=batch_norm, drop_out=drop_out),
            conv_block(32, 64, 3, stride=2, relu_alpha=relu_alpha,  batch_norm=batch_norm, drop_out=drop_out),
            conv_block(64, 128, 3, stride=2, relu_alpha=relu_alpha,  batch_norm=batch_norm, drop_out=drop_out),
            conv_block(128, 256, 3, stride=2, relu_alpha=relu_alpha,  batch_norm=batch_norm, drop_out=drop_out),
            conv_block(256, 512, 3, stride=2, relu_alpha=relu_alpha, batch_norm=batch_norm, drop_out=drop_out),
            conv_block(512, 1024, 3, stride=2, relu_alpha=relu_alpha, batch_norm=batch_norm, drop_out=drop_out),
            conv_block(1024, 2048, 3, stride=2, relu_alpha=relu_alpha, batch_norm=batch_norm, drop_out=drop_out),
            #nn.AvgPool3d(kernel_size=(1, 8, 8), stride=(1, 8, 8)),
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Flatten()
        )
        self.fc_layer = nn.Linear(2048, 1)  # Reduce feature dimension to 1
        self.output_activation = nn.Sigmoid()

            
    def forward(self, x):
        batch_size, seq_length, channels, height, width = x.size()
        x = x.reshape(batch_size * seq_length, channels, height, width)  # Change this line
        if hasattr(self, 'padding'):
            x = F.pad(x, self.padding, 'constant', 0)
        for idx, layer in enumerate(self.conv_model):
            x = layer(x)
            #print(f"Size after layer {idx + 1}: {x.size()}")
        x = self.fc_layer(x)  # Fully connected layer to reduce feature dimension
        x = self.output_activation(x)
        x = x.view(batch_size, seq_length, -1) 
        return x
    
    