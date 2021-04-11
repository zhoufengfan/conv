import torch
import torch.nn as nn

if __name__ == '__main__':
    # m = nn.Conv1d(in_channels=16, out_channels=33, kernel_size=3, stride=2)
    # out_channels can be set to any int value(if not overflow).
    m = nn.Conv1d(in_channels=1, out_channels=10000, kernel_size=3, stride=1)
    # [batch_size, in_channels, $L_{in}$]
    # input = torch.randn(20, 16, 50)
    input = torch.randn(20, 1, 5)
    print("input.shape is", input.shape)
    output = m(input)
    print("output.shape is", output.shape)
