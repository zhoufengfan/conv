import torch
import torch.nn as nn

if __name__ == '__main__':
    m = nn.Conv1d(in_channels=16, out_channels=33, kernel_size=3, stride=2)
    input = torch.randn(20, 16, 50)
    output = m(input)
    print("output.shape is", output.shape)
