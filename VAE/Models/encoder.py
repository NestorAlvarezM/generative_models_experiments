import torch
import torch.nn as nn
from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter

class Encoder(nn.Module):
    def __init__(self, input_channels,input_shape, embedding_dim):
        super(Encoder, self).__init__()
        self.input_channels = input_channels
        self.embedding_dim = embedding_dim
        self.input_width=input_shape[2]
        self.input_height=input_shape[3]
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), #TODO should i use Relu inplace?
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512,128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32,16,kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x,start_dim=1)

        return x

def test_encoder(encoder, input_shape, embedding_dim):
    batch_size, channels, width, height = input_shape

    # Generate a random input tensor with the given shape
    input_tensor = torch.randn(batch_size, channels, width, height)
    # Pass the input tensor through the encoder
    output_tensor = encoder(input_tensor)
    # Create a computation graph and visualize it
    output = encoder(input_tensor)
    # Check if the output shape matches the desired shape
    desired_output_shape = (batch_size, embedding_dim)
    assert output_tensor.shape == desired_output_shape, f"Expected shape {desired_output_shape}, but got {output_tensor.shape}"
    writer = SummaryWriter()

    # Write the model graph to TensorBoard
    writer.add_graph(encoder, input_tensor)

    # Close the writer to flush all the data to disk
    writer.close()

    print("Test passed! given the output with shape",input_shape,"you get a tensor of output shape",desired_output_shape)

# Assuming you've already defined your encoder with input_channels and embedding_dim
input_shape = (8, 3, 224, 224)  # Example input shape: 32 samples, 3 channels, 64x64 image size
embedding_dim= 16
encoder=Encoder(3,input_shape,16)
test_encoder(encoder, (1,3,224,224), 36)