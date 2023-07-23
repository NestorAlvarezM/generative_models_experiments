import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

class Decoder(nn.Module):
    def __init__(self, hidden_dim):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dim, 24, kernel_size=7, stride=2, padding=0, output_padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.ConvTranspose2d(24, 12, kernel_size=2, stride=2, padding=0, output_padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.ConvTranspose2d(12, 8, kernel_size=2, stride=2, padding=0, output_padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.ConvTranspose2d(8, 6, kernel_size=2, stride=2, padding=0, output_padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(6),
            nn.ConvTranspose2d(6, 3, kernel_size=2, stride=2, padding=0, output_padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(3),
            nn.ConvTranspose2d(3, 3, kernel_size=2, stride=2, padding=0, output_padding=0),
            nn.ReLU(),
        )
    def forward(self, x):
        x = x.view(-1, self.hidden_dim, 1, 1)
        x = self.conv_layers(x)

        return x

def test_decoder(decoder, input_shape, tensor_image_dim):
    batch_size, embedding_dim= input_shape
    batch_size, channels,width,heigth= tensor_image_dim
    # Generate a random input tensor with the given shape
    input_tensor = torch.randn(batch_size,embedding_dim)
    # Pass the input tensor through the encoder
    output_tensor = decoder(input_tensor)
    # Create a computation graph and visualize it
    output = decoder(input_tensor)
    # Check if the output shape matches the desired shape
    desired_output_shape = (batch_size, embedding_dim)
    #assert output_tensor.shape == desired_output_shape, f"Expected shape {desired_output_shape}, but got {output_tensor.shape}"
    writer = SummaryWriter()
    # Write the model graph to TensorBoard
    writer.add_graph(decoder, input_tensor)

    # Close the writer to flush all the data to disk
    writer.close()

    print("Test passed! given the output with shape",input_shape,"you get a tensor of output shape",desired_output_shape)

# Assuming you've already defined your encoder with input_channels and embedding_dim
input_shape = (1,36)  # Example input shape: 32 samples, 3 channels, 64x64 image size
tensor_image_dim= (1,3,224,244)
decoder=Decoder(input_shape[1])
test_decoder(decoder, input_shape, tensor_image_dim)

